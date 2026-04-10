import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
import cv2
import numpy as np
from google.protobuf import wrappers_pb2

# Boston Dynamics SDK (僅用於視覺與資料處理，不搶 Lease)
import bosdyn.client
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2, network_compute_bridge_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.client.robot_command import RobotCommandBuilder

# ROS 2 轉換與 Action 訊息
from bosdyn_msgs.conversions import convert
from spot_msgs.action import Manipulation
from spot_msgs.action import RobotCommand

import threading
import time

#ros2
from geometry_msgs.msg import Twist

class SpotFetchROS2Node(Node):
    def __init__(self):
        super().__init__('spot_fetch_ros2_node')
        
        # --- 1. ROS 2 設定 ---
        # 建立 Manipulation Action Client (確認 namespace 是否需要加上 robot_name，例如 '/spot/manipulation')
        self.manip_client = ActionClient(self, Manipulation, '/manipulation')
        self.robot_client = ActionClient(self, RobotCommand, '/robot_command')
        
        # --- 2. SDK 設定 (不要求 Lease) ---
        self.get_logger().info('正在設定robot...')
        self.get_logger().info("正在連線至 Spot SDK (僅讀取影像/NCS)...")
        sdk = bosdyn.client.create_standard_sdk('SpotFetchROS2')
        sdk.register_service_client(NetworkComputeBridgeClient)
        # TODO: 請替換為你的機器人 IP 與帳密
        # self.robot = sdk.create_robot("10.0.0.3") 
        self.robot = sdk.create_robot("192.168.80.3") 
        self.robot.authenticate("admin", "eqyqp33u8i74")
        self.robot.time_sync.wait_for_sync()
        self.ncb_client = self.robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
        
        # --- 3. 任務參數 ---
        self.ml_service = "fetch-server" # TODO: 填入你的服務名稱
        self.model_name = "best.engine"       # TODO: 填入你的模型名稱
        self.target_label = "Bottle_and_Can"               # TODO: 你要找的標籤
        self.min_confidence = 0.5
        
        # 啟動主迴圈計時器 (每 2 秒執行一次偵測)
        self.timer = self.create_timer(0.5, self.fetch_loop)
        self.is_fetching = False # 避免重複觸發
        self.is_approaching = False
        self.move_msg = Twist()
        #-----------------------walk--------------------------------------------------
        # 訂閱nav速度指令
        self.nav_sub = self.create_subscription(Twist, 'cmd_vel_nav', self.nav_callback, 10)
        # 發布最終給 Spot 的控制速度 (spot_driver 訂閱這個)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 狀態變數
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vrot = 0.0

    def nav_callback(self, msg):
        """
        核心邏輯：如果沒在夾東西，就把導航的速度直接『搬運』給 Spot。
        如果正在夾東西，則不轉發（或者發布 0,0,0 確保靜止）。
        """
        if not self.is_fetching and not self.is_approaching:
            self.cmd_vel_pub.publish(msg)
        elif self.is_approaching:
            self.cmd_vel_pub.publish(self.move_msg)
        else:
            # 如果正在處理物品，我們不主動發布 0，
            # 讓 fetch_loop 控制速度就好，避免頻率衝突
            pass
    
    def fetch_loop(self):
        if self.is_fetching:
            return
        
        self.get_logger().info('正在透過 NCS 搜尋物體...')
        
        # 取得影像與辨識結果
        target_obj, image_full, vision_tform_obj = self.get_obj_and_img(
            ['frontleft_fisheye_image', 'frontright_fisheye_image']
        )

        if target_obj is None or vision_tform_obj is None:
            if self.is_approaching:
                self.get_logger().info("目標遺失，恢復巡邏模式...")
                self.is_approaching = False
            return 

        # 2. 計算距離 (Vision Frame 的原點通常在機器人啟動點，
        # 但 vision_tform_obj.position 是物體在該座標系的位置)
        # 如果要精確計算「相對於機器人目前中心」的距離：
        # dist_x = vision_tform_obj.x
        # dist_y = vision_tform_obj.y
        # dist_z = vision_tform_obj.z
        
        # # 計算 3D 距離 (你也可以只計算平面距離 x, y)
        # distance = math.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        try:
            # 2. 【核心修正】將 Vision Frame 的座標轉換為 Body Frame
            # 我們需要「現在」機器人身體相對於 Vision Frame 的位置
            # image_full.shot.transforms_snapshot 包含了機器人目前的狀態
            vision_tform_body = frame_helpers.get_a_tform_b(
                image_full.shot.transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.BODY_FRAME_NAME
            )

            # 計算物體相對於身體的座標
            # 原理：body_pose = (vision_tform_body)^-1 * vision_tform_obj
            body_tform_obj = vision_tform_body.inverse() * vision_tform_obj
            
            tx = body_tform_obj.x
            ty = body_tform_obj.y
            distance = math.sqrt(tx**2 + ty**2)
            angle_to_target = math.atan2(ty, tx)
            
            self.get_logger().info(f"相對座標: 前方={tx:.2f}m, 左方={ty:.2f}m")

        except Exception as e:
            self.get_logger().error(f"座標轉換失敗: {e}")
            return
        
        self.get_logger().info(f'偵測到目標，距離: {distance:.2f} 公尺')

        # 狀況 A：太遠 (超過 10m)
        if distance > 8.0:
            self.get_logger().info("目標太遠，忽略中...")
            return

        # 狀況 B：在 2m 到 10m 之間 -> 慢慢走過去
        elif 2.0 < distance <= 8.0:
            self.get_logger().info(f"接近目標中... (距離 {distance:.2f}m)")
            self.is_approaching = True
            
            # 簡單的比例控制 (P Control) 讓它面向並靠近物體
            # self.move_msg = Twist()
            
            # 設定前進速度 (限制在 0.3 m/s 比較慢、比較安全)
            self.move_msg.linear.x = 0.3 
            
            # 設定轉向速度：讓機器人正對物體
            self.move_msg.angular.z = angle_to_target * 0.5 # 0.5 是轉向增益
            
            # 發布速度指令給 spot_driver
            # self.cmd_vel_pub.publish(self.move_msg)
            return # 結束本次 loop，等下一次 2秒後的 timer 再判斷距離

        # 狀況 C：距離 2m 以內 -> 停止並夾取
        else:
            self.get_logger().info("已抵達目標範圍，開始夾取程序...")
            self.is_approaching = False # 結束趨近狀態
            self.is_fetching = True    # 鎖定夾取狀態
            
            # 先發布停止指令
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)

        
        # # # 3. 距離判斷：超過 2.0 公尺就只紀錄但不撿取
        # # if distance > 2.5:
        # #     self.get_logger().warn(f'目標太遠 ({distance:.2f}m > 2.0m)，繼續巡邏中...')
        # #     return

        # # # 4. 符合條件，開始夾取
        # # self.get_logger().info(f'🎉 目標在範圍內! 準備執行夾取...')
 
        # # self.is_fetching = True 

        # ==========================================
        # 📍 這裡是你未來可以加入 Nav2 邏輯的地方：
        # 1. 呼叫 Nav2 暫停目前巡邏
        # 2. 算好距離，呼叫 Nav2 靠近 (取代原本的 se2_trajectory_command)
        # ==========================================

        # 計算像素中心
        center_px_x, center_px_y = self.find_center_px(target_obj.image_properties.coordinates)

        # 構造 SDK 夾取指令
        pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image_full.shot.transforms_snapshot,
            frame_name_image_sensor=image_full.shot.frame_name_image_sensor,
            camera_model=image_full.source.pinhole
        )
        grasp.grasp_params.grasp_palm_to_fingertip = 0.6
        grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

        manip_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # --- 將SDK夾取指令轉換並傳送給 ROS 2 ---
        self.send_ros2_manipulation_goal(manip_request)

    def send_cmd_blocking(self, sdk_cmd, label):
        """同步阻塞式發送：確保前一個動作完成才回傳"""
        if not self.robot_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'無法連接到 Action Server，請確認 spot_driver 狀態')
            return False    

        goal_msg = RobotCommand.Goal()
        convert(sdk_cmd, goal_msg.command)
        
        self.get_logger().info(f'▶執行步驟: {label}')
        
        # 發送目標並等待接受
        send_goal_future = self.robot_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_goal_future.done():
            time.sleep(0.1) # 讓出 CPU 給其他任務
            
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'{label} 被機器人拒絕')
            return False

        # 等待執行結果
        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            time.sleep(0.1) # 讓出 CPU 給其他任務
        self.get_logger().info(f'{label} 完成')
        return True

    
    def send_ros2_manipulation_goal(self, manip_request):
        if not self.manip_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("找不到 Manipulation Action Server!")
            self.is_fetching = False
            return

        goal_msg = Manipulation.Goal()
        # 把 SDK 請求轉換為 ROS 2 Goal 訊息
        convert(manip_request, goal_msg.command)

        self.get_logger().info('正在發送 ROS 2 夾取指令...')
        send_goal_future = self.manip_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('夾取請求被拒絕！')
            self.is_fetching = False
            return

        self.get_logger().info('夾取請求已接受，執行中...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        # GoalStatus.STATUS_SUCCEEDED 通常是 4
        if status == 4: 
            self.get_logger().info('✅ 夾取動作確認成功！')
            action_thread = threading.Thread(target=self.post_grasp_sequence)
            action_thread.start()
        else:
            self.get_logger().error(f'❌ 夾取失敗，狀態碼: {status}')
            self.is_fetching = False # 重置狀態，讓它繼續偵測
        # # 動作完成後的回調
        # result = future.result().result
        # self.get_logger().info('✅ 夾取動作結束！ 回收進背上垃圾桶')
        # action_thread = threading.Thread(target=self.post_grasp_sequence)
        # action_thread.start()
        
    def post_grasp_sequence(self):
        
        # 1. 展開手臂
        self.send_cmd_blocking(RobotCommandBuilder.arm_ready_command(), "手臂預備 (Ready)")
        
        try:
            
            # 2. 三段式移動路徑
            poses = [
                (0.25, 0.2, 0.6, "安全點 1: 側上方"),
                (0.0, 0.0, 0.7, "安全點 2: 正上方"),
                (-0.2, 0.0, 0.6, "安全點 3: 背部後方")
            ]

            for x, y, z, label in poses:    
                cmd = RobotCommandBuilder.arm_pose_command(
                    x, y, z, 0.707, 0, 0.707, 0, 
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, 
                    seconds=1.5
                )
                self.send_cmd_blocking(cmd, label)

            # 3. 開啟夾爪
            self.send_cmd_blocking(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0), "開啟夾爪")
            time.sleep(1.0)

            # 4. 收回手臂 （可加入判斷有無下個夾取物）
            self.send_cmd_blocking(RobotCommandBuilder.arm_stow_command(), "手臂收納 (Stow)")
            
            # 5.關閉夾爪
            self.send_cmd_blocking(RobotCommandBuilder.claw_gripper_open_fraction_command(0.0), "關閉夾爪")
            time.sleep(1.0)
            
        except Exception as e:
            self.get_logger().error(f"執行回收動作時出錯: {e}")
        finally:
            self.is_fetching = False # 動作全做完了，才允許下一次偵測
            # # 5. 最後讓它坐下
            # self.send_cmd_blocking(RobotCommandBuilder.synchro_sit_command(), "任務完成，坐下休息")
            
            # self.get_logger().info('🎊 所有自動化動作執行完畢，Spot 已安全坐下！')
            
            # ==========================================
            # 📍 這裡是你未來可以加入的收尾邏輯：
            # 1. 發送 RobotCommand 把手臂收到背後 (arm_stow)
            # 2. 呼叫 Nav2 繼續巡邏
            # ==========================================
            
            # 恢復狀態，準備尋找下一個
            # self.is_fetching = False

    # ---------------------------------------------------------
    # 以下為保留自原版 SDK 腳本的工具函式 (無需修改核心邏輯)
    # ---------------------------------------------------------
    def get_obj_and_img(self, image_sources):
        for source in image_sources:
            img_src = network_compute_bridge_pb2.ImageSourceAndService(image_source=source)
            input_data = network_compute_bridge_pb2.NetworkComputeInputData(
                image_source_and_service=img_src, 
                model_name=self.model_name,
                min_confidence=self.min_confidence, 
                rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL
            )
            server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(service_name=self.ml_service)
            process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(input_data=input_data, server_config=server_data)

            try:
                resp = self.ncb_client.network_compute_bridge_command(process_img_req)
            except Exception as e:
                self.get_logger().error(f'NCS 連線錯誤: {e}')
                return None, None, None

            best_obj = None
            highest_conf = 0.0
            best_vision_tform_obj = None

            if len(resp.object_in_image) > 0:
                for obj in resp.object_in_image:
                    obj_label = obj.name.split('_label_')[-1]
                    if obj_label != self.target_label:
                        continue
                        
                    conf_msg = wrappers_pb2.FloatValue()
                    obj.additional_properties.Unpack(conf_msg)
                    conf = conf_msg.value

                    try:
                        vision_tform_obj = frame_helpers.get_a_tform_b(
                            obj.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
                            obj.image_properties.frame_name_image_coordinates)
                    except:
                        vision_tform_obj = None

                    if conf > highest_conf and vision_tform_obj is not None:
                        highest_conf = conf
                        best_obj = obj
                        best_vision_tform_obj = vision_tform_obj

            if best_obj is not None:
                return best_obj, resp.image_response, best_vision_tform_obj

        return None, None, None

    def find_center_px(self, polygon):
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for vert in polygon.vertexes:
            if vert.x < min_x: min_x = vert.x
            if vert.y < min_y: min_y = vert.y
            if vert.x > max_x: max_x = vert.x
            if vert.y > max_y: max_y = vert.y
        x = math.fabs(max_x - min_x) / 2.0 + min_x
        y = math.fabs(max_y - min_y) / 2.0 + min_y
        return (x, y)

def main(args=None):
    rclpy.init(args=args)
    node = SpotFetchROS2Node()
    
    # 關鍵修正：使用多執行緒執行器
    # 這允許一個執行緒在 post_grasp_sequence 裡等待動作完成時，
    # 另一個執行緒還能處理來自 Action Server 的回傳訊息
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('正在關閉節點...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
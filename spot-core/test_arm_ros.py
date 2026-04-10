import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# 匯入 Boston Dynamics SDK 的 Protobuf 定義 (只用來建立格式，不實際連線)
from bosdyn.api import manipulation_api_pb2, geometry_pb2

# 匯入 ROS 2 轉換工具與 Action
from bosdyn_msgs.conversions import convert
from spot_msgs.action import Manipulation

class SimpleManipTestNode(Node):
    def __init__(self):
        super().__init__('simple_manip_test_node')
        
        # 建立 Action Client (如果你的 namespace 有改，例如 '/spot/manipulation'，請自行調整)
        self.client = ActionClient(self, Manipulation, '/manipulation')

    def send_test_goal(self):
        self.get_logger().info('等待 Manipulation Action Server...')
        self.client.wait_for_server()

        # --- 1. 建立一個簡單的 SDK Protobuf 指令 ---
        # 假裝我們要抓一個機器人正前方 0.8m，上方 0.4m 的東西
        pick_cmd = manipulation_api_pb2.PickObject(
            frame_name="body", # 使用機器人本身的身體座標系
            object_rt_frame=geometry_pb2.Vec3(x=0.8, y=0.0, z=0.4)
        )

        # 包裝成 Request
        request_proto = manipulation_api_pb2.ManipulationApiRequest(
            pick_object=pick_cmd
        )

        # --- 2. 轉換為 ROS 2 Action Goal ---
        goal_msg = Manipulation.Goal()
        # 注意：根據你剛才的介面，Action 的第一個欄位叫 'command'
        convert(request_proto, goal_msg.command)

        # --- 3. 發送給 spot_driver ---
        self.get_logger().info('發送測試夾取指令 (正前方 80cm)...')
        self.send_goal_future = self.client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        # 接收手臂執行時的即時回報
        # 取出 current_state 的整數值
        state_val = feedback_msg.feedback.feedback.current_state.value
        self.get_logger().info(f'🤖 手臂即時狀態碼: {state_val}')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('目標被驅動程式拒絕 :(')
            return

        self.get_logger().info('目標已接受，等待執行結果...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'✅ 測試結束！是否成功: {result.success}, 訊息: {result.message}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = SimpleManipTestNode()
    node.send_test_goal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        self.get_logger().info("手動中斷。")

if __name__ == '__main__':
    main()
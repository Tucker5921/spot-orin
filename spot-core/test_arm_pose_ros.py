import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import threading
import time

# SDK 與 ROS 2 轉換工具
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client import frame_helpers
from spot_msgs.action import RobotCommand
from bosdyn_msgs.conversions import convert

class SpotArmAutomation(Node):
    def __init__(self):
        super().__init__('spot_arm_automation')
        # 根據你的 action list，路徑確核為 /robot_command
        self.client = ActionClient(self, RobotCommand, '/robot_command')
        self.get_logger().info('🤖 手機自動化序列已啟動，等待 Action Server...')

    def send_cmd_blocking(self, sdk_cmd, label):
        """同步阻塞式發送：確保前一個動作完成才回傳"""
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'❌ 無法連接到 Action Server，請確認 spot_driver 狀態')
            return False

        goal_msg = RobotCommand.Goal()
        convert(sdk_cmd, goal_msg.command)
        
        self.get_logger().info(f'▶️ 執行步驟: {label}')
        
        # 發送目標並等待接受
        send_goal_future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'❌ {label} 被機器人拒絕')
            return False

        # 等待執行結果
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info(f'✅ {label} 完成')
        return True

    def run_mission(self):
        # 1. 確保站立
        self.send_cmd_blocking(RobotCommandBuilder.synchro_stand_command(), "自動站立")
        time.sleep(1.0)

        # 2. 展開手臂
        self.send_cmd_blocking(RobotCommandBuilder.arm_ready_command(), "手臂預備 (Ready)")
        
        # 確保夾爪關緊 (例如夾住垃圾後) ---
        self.send_cmd_blocking(RobotCommandBuilder.claw_gripper_close_command(), "關閉夾爪")
        time.sleep(0.5) # 給夾爪一點點物理閉合的時間

        # 3. 三段式移動路徑
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

        # 4. 開啟夾爪
        self.send_cmd_blocking(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0), "開啟夾爪")
        time.sleep(1.0)

        # 5. 收回手臂
        self.send_cmd_blocking(RobotCommandBuilder.arm_stow_command(), "手臂收納 (Stow)")
        time.sleep(2.0)

        # 6. 最後讓它坐下
        self.send_cmd_blocking(RobotCommandBuilder.synchro_sit_command(), "任務完成，坐下休息")
        
        self.get_logger().info('🎊 所有自動化動作執行完畢，Spot 已安全坐下！')

def main(args=None):
    rclpy.init(args=args)
    node = SpotArmAutomation()
    
    # 執行任務
    try:
        node.run_mission()
    except Exception as e:
        print(f"執行中發生錯誤: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
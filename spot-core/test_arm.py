import time
import bosdyn.client
from bosdyn.client import frame_helpers
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive

sdk = bosdyn.client.create_standard_sdk('SpotFetchClient')
robot = sdk.create_robot("10.0.0.3")
robot.authenticate("admin", "eqyqp33u8i74")

# Time sync is necessary so that time-based filter requests can be converted
robot.time_sync.wait_for_sync()

command_client = robot.ensure_client(RobotCommandClient.default_service_name)
lease_client = robot.ensure_client(LeaseClient.default_service_name)

#拿lease
lease_client.take()
with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

    #0. 站起來
    time.sleep(1)
    a = input("press enter to stand")
    stand_cmd = RobotCommandBuilder.synchro_stand_command()
    command_client.robot_command(stand_cmd)
    time.sleep(2)
    a = input("press enter to ready")

    # 1. 展開手臂到預備姿勢 (Ready)
    ready_cmd = RobotCommandBuilder.arm_ready_command()
    command_client.robot_command(ready_cmd)
    time.sleep(2)
    a = input("press enter to arm_pose")

    # 2. 假設容器口在背部後方，先移動到上方 0.3m 處
    safe_x, safe_y, safe_z = 0.25, 0.2, 0.6  # 相對於 body 框
    arm_safe_cmd = RobotCommandBuilder.arm_pose_command(
        safe_x, safe_y, safe_z, 0.707, 0, 0.707, 0, 
        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, seconds=2.0)
    command_client.robot_command(arm_safe_cmd)
    time.sleep(2) # 等待到達安全位置
    safe_x, safe_y, safe_z = 0.0, 0.0, 0.7  # 相對於 body 框
    arm_safe_cmd = RobotCommandBuilder.arm_pose_command(
        safe_x, safe_y, safe_z, 0.707, 0, 0.707, 0, 
        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, seconds=2.0)
    command_client.robot_command(arm_safe_cmd)
    time.sleep(2) # 等待到達安全位置
    safe_x, safe_y, safe_z = -0.2, 0.0, 0.6  # 相對於 body 框
    arm_safe_cmd = RobotCommandBuilder.arm_pose_command(
    safe_x, safe_y, safe_z, 0.707, 0, 0.707, 0, 
    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, seconds=2.0)
    command_client.robot_command(arm_safe_cmd)
    time.sleep(2) # 等待到達安全位置
    a = input("press enter to stow")

    # 3. 收納手臂 (Stow)
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    command_client.robot_command(stow_cmd)
    time.sleep(2)
    a = input("press enter to sit")

    sit_cmd = RobotCommandBuilder.synchro_sit_command()
    command_client.robot_command(sit_cmd)
    time.sleep(2)
    a = input("press enter to end")

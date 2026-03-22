from URBasic import robotModel, urScriptExt
import numpy as np
import time
import keyboard  # Make sure this is installed
import GripperFunctions as gripper
import pandas as pd
import os

# Initialize robot
robot_ip = '192.168.1.10'  # ← Change to your IP
rtde_conf_file = 'URBasic/rtdeConfigurationDefault.xml'

model = robotModel.RobotModel()
robot = urScriptExt.UrScriptExt(host=robot_ip, robotModel=model, conf_filename=rtde_conf_file)

gripper.open_gripper(robot)
grip_open = True

robot.init_realtime_control()

# Start from current pose
pose = robot.get_actual_tcp_pose()

latency_records = []

# custom limits
# x

# y
base = 0.225
# z
rolling_table = -0.2955
utility_box = -0.653

# -----Choose Surface to Exclude----- (None for no limits)
custom_x_limit = None
custom_y_limit = base
custom_z_limit = rolling_table

# Movement step size (meters)
step = 0.002

print("Use arrow keys to move the robot (X/Z). Press ESC to quit.")

try:
    while True:

        keyboard_click_time = time.time()
        updated = False

        if keyboard.is_pressed('w') or keyboard.is_pressed('up'):
            pose[2] += step  # Z+
            updated = True
        if keyboard.is_pressed('s') or keyboard.is_pressed('down'):
            pose[2] -= step  # Z-
            updated = True
        if keyboard.is_pressed('a') or keyboard.is_pressed('left'):
            pose[0] -= step  # X-
            updated = True
        if keyboard.is_pressed('d') or keyboard.is_pressed('right'):
            pose[0] += step  # X+
            updated = True
        if keyboard.is_pressed('q') or keyboard.is_pressed('alt'):
            pose[1] -= step  # X-
            updated = True
        if keyboard.is_pressed('e') or keyboard.is_pressed('shift'):
            pose[1] += step  # X+
            updated = True
        if keyboard.is_pressed('esc'):
            print("Exiting control loop.")
            break

        if updated:
            robot.set_realtime_pose(pose)
            print(f"Moved to: {np.round(pose, 3)}")
            time.sleep(0.05)  # Prevent overloading command buffer

        robot_command_time = time.time()

        if keyboard.is_pressed('space'):
            if not grip_open:
                gripper.open_gripper(robot)
                custom_z_limit = -0.2955
                grip_open = True
                robot.init_realtime_control()
            else:
                gripper.close_gripper(robot)
                custom_z_limit = pose[2]
                grip_open = False
                robot.init_realtime_control()
        robot_motion_end_time = time.time()
        if updated:
            pose[2] = max(pose[2], custom_z_limit)
            gesture_to_command_latency = robot_command_time - keyboard_click_time
            command_to_motion_latency = robot_motion_end_time - robot_command_time
            total_latency = robot_motion_end_time - keyboard_click_time
            latency_records.append((gesture_to_command_latency, command_to_motion_latency, total_latency))

        if keyboard.is_pressed('shift'):
            was_paused = True
            print("Saving trajectory data to Excel...")

            latency_df = pd.DataFrame(latency_records,
                                      columns=['Gesture to command', 'Command to motion', 'Gesture to motion'])

            # Set your target folder
            folder_path = r"C:\Users\Lydia\Undergraduate Research w Dr. Korivand\trials\excel sheets from trials"
            os.makedirs(folder_path, exist_ok=True)  # Ensure folder exists

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(folder_path, f"Keyboard Latency_{timestamp}.xlsx")

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                latency_df.to_excel(writer, sheet_name='Latency Summary', index=False)

            print(f"✅ Saved: {filename}")

except KeyboardInterrupt:
    print("Stopped by user.")

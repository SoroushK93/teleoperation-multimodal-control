import cv2
import mediapipe as mp
from URBasic import robotModel, urScriptExt
from Gestures import gestures  # ← Import gesture recognition
import numpy as np
import time
import GripperFunctions as gripper
from scipy.spatial.transform import Rotation as R
import keyboard
import pandas as pd
import os


# === CONFIG ===
frame_width = 640
frame_height = 480

# === OPENCV CAMERA SETUP ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam.")

# === MEDIAPIPE SETUP ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

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
custom_z_limit = utility_box

# === ROBOT SETUP ===
robot_ip = '192.168.1.10'  # ← Replace with your robot's IP
rtde_conf_file = 'URBasic/rtdeConfigurationDefault.xml'

model = robotModel.RobotModel()
robot = urScriptExt.UrScriptExt(host=robot_ip, robotModel=model, conf_filename=rtde_conf_file)
robot.init_realtime_control()

# === Reset robot to face-down position at start ===
start_pose = robot.get_actual_tcp_pose()
face_down_pose = [start_pose[0],start_pose[1], start_pose[2], 3.14, 0.0, 0.0]  # change orientation to be face down
print('=== Moving to Face Down ===')
robot.movej(robot.get_inverse_kin(face_down_pose), 0.5, 0.5)
pose = face_down_pose.copy()
robot.init_realtime_control()

latency_records = []
was_paused = True

paused = False

gesture_timers = {
    'rocknroll': 0.0,
    'down rocknroll': 0.0
}
GESTURE_HOLD_THRESHOLD = 0.50 # sec

def update_gesture_timer(gesture_name, is_detected):
    current_time = time.time()
    if is_detected:
        if gesture_timers[gesture_name] == 0.0:
            gesture_timers[gesture_name] = current_time
        elif current_time - gesture_timers[gesture_name] > GESTURE_HOLD_THRESHOLD:
            gesture_timers[gesture_name] = 0.0
            return True
    else:
        gesture_timers[gesture_name] = 0.0
    return False

def apply_tool_z_rotation(rotation_vector, delta_rz):
    """Rotate current TCP orientation around tool Z-axis by delta_rz radians."""
    current_rot = R.from_rotvec(rotation_vector)
    local_z_rot = R.from_euler('z', delta_rz)
    new_rot = current_rot * local_z_rot  # Tool frame rotation
    return new_rot.as_rotvec()

# === MOTION SETUP ===
step = 0.003

# === MAIN LOOP ===
try:
    while True:
        ret, frame = cap.read()
        hand_movement_time = time.time()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if paused and was_paused is False:
            was_paused = True
            print("Saving trajectory data to Excel...")

            latency_df = pd.DataFrame(latency_records,
                                      columns=['Gesture to command', 'Command to motion', 'Gesture to motion'])

            # Set your target folder
            folder_path = r"C:\Users\Lydia\Undergraduate Research w Dr. Korivand\trials\excel sheets from trials"
            os.makedirs(folder_path, exist_ok=True)  # Ensure folder exists

            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(folder_path, f"Gesture Latency_{timestamp}.xlsx")

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                latency_df.to_excel(writer, sheet_name='Latency Summary', index=False)

            print(f"✅ Saved: {filename}")

        if not paused:
            if was_paused:  # i.e., just resumed
                was_paused = False
        else:
            was_paused = True

        if keyboard.is_pressed('space'):
            paused = not paused
            print("Paused" if paused else "Resumed")
            while keyboard.is_pressed('space'):
                pass

        updated = False

        if results.multi_hand_landmarks and results.multi_handedness:
            rocknroll_detected = False
            down_rocknroll_detected = False
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                gesture_result = gestures(landmarks, label)

                # If returned as (gesture, handedness), extract just gesture string
                if isinstance(gesture_result, tuple):
                    gesture = gesture_result[0]
                else:
                    gesture = gesture_result

                y_offset = 50 if label == "Right" else 200
                cv2.putText(frame, f'{label}: {gesture}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # === GESTURE → ROBOT ACTION ===
                if not paused:
                    if gesture == "thumbs up":
                        pose[2] += step  # Z+
                        updated = True
                    elif gesture == "thumbs down":
                        pose[2] -= step  # Z-
                        updated = True
                    elif gesture == "point left":
                        pose[1] -= step  # X-
                        updated = True
                    elif gesture == "point right":
                        pose[1] += step  # X+
                        updated = True
                    elif gesture == "point up":
                        pose[0] -= step  # Y+
                        updated = True
                    elif gesture == "point down":
                        pose[0] += step  # Y-
                        updated = True
                    elif gesture == "rocknroll":
                        if update_gesture_timer('rocknroll', True):
                            gripper.open_gripper(robot)
                            custom_z_limit = -0.2955
                            time.sleep(1)
                            robot.init_realtime_control()
                            updated = True
                    elif gesture == "down rocknroll":
                        if update_gesture_timer('down rocknroll', True):
                            gripper.close_gripper(robot)
                            custom_z_limit = pose[2]
                            time.sleep(1)
                            robot.init_realtime_control()
                            updated = True
                    elif gesture == 'pinky down':
                        pose[3:6] = apply_tool_z_rotation(pose[3:6], 0.05)
                        updated = True
                    elif gesture == 'pinky':
                        pose[3:6] = apply_tool_z_rotation(pose[3:6], -0.05)
                        updated = True

        if paused:
            updated = False
        if updated:
            pose[2] = max(pose[2], custom_z_limit)
            robot.set_realtime_pose(pose)
            time.sleep(0.05)
        robot_command_time = time.time()

        if paused:
            cv2.line(frame, (200, 60), (435, 60), (0, 0, 255), 50)
            cv2.putText(frame, "Motion Paused", (200, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        robot_motion_end_time = time.time()
        if not paused and updated:
            gesture_to_command_latency = robot_command_time - hand_movement_time
            command_to_motion_latency = robot_motion_end_time - robot_command_time
            total_latency = robot_motion_end_time - hand_movement_time
            latency_records.append((gesture_to_command_latency, command_to_motion_latency, total_latency))

        cv2.imshow("Webcam + Gesture Robot Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

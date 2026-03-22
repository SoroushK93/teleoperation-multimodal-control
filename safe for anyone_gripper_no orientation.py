import cv2
import mediapipe as mp
from URBasic import robotModel, urScriptExt
from Gestures import gestures
import numpy as np
import math
import time
import keyboard
from scipy.spatial.transform import Rotation as R
import GripperFunctions as gripper
from PIL import Image, ImageDraw, ImageFont

"""
Dual-camera motion.
Summary:
→ The safest and most deliberate version so far — gesture-gated motion ensures
user intent, supports both hands at once, and prevents false movements.
Best for precise, stable control with low error risk.
"""


# === Robot setup ===
robot_ip = '192.168.1.10'
rtde_conf_file = 'URBasic/rtdeConfigurationDefault.xml'

model = robotModel.RobotModel()
robot = urScriptExt.UrScriptExt(host=robot_ip, robotModel=model, conf_filename=rtde_conf_file)

# === Reset robot to face-down position at start ===
start_pose = robot.get_actual_tcp_pose()
face_down_pose = [-0.140,1.040, -0.1, 3.14, 0.0, 0.0]  # change orientation to be face down
print('=== Moving to Face Down ===')
robot.movej(robot.get_inverse_kin(face_down_pose), 0.5, 0.5)
pose = face_down_pose.copy()
robot.init_realtime_control()

cap_main = None
cap_depth = None

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

if cap1.isOpened():
    cap_main = cap1
    cap_depth = cap0 if cap0.isOpened() else False
elif cap0.isOpened():
    cap_main = cap0
    cap_depth = False
else:
    raise RuntimeError("No usable webcam found.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.75, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

prev_pos_main = None
prev_pos_depth = None
prev_landmarks_dict = {}

recovery_frames_remaining = 0
step = 0.002
scale = 0.7
paused = False
joint_limit_blocked = False
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

gesture_timers = {
    'rocknroll': 0.0,
    'down rocknroll': 0.0
}
GESTURE_HOLD_THRESHOLD = 0.50 # sec

MAX_POS_DELTA = 0.03
MAX_ROT_DELTA = 0.1

controls_text = [
    ["____ Controls: ____", "____ Motion: ____","____ Hand:Robot ____"],
    ["⌨Spacebar: Pause/Resume", "✊Fist: No Motion", "up/down: up/down"],
    ["🤘: Open Gripper", "🤚Open Palm: Motion", "left/right: left/right"],
    ["👇🤘: Close Gripper   (rocknroll down)", "","forward/back: forward/back"],
    ["Once hand leaves the frame, use fist to reset.     Check frame for hand in both cameras", "", ""]
]

control_window_name = "Controls"

def show_control_window():
    width = 1000
    height = 31 * len(controls_text)
    col_width = width // 3

    # Create a blank image using PIL
    image = Image.new("RGB", (width, height), (50, 50, 175))
    draw = ImageDraw.Draw(image)

    # Try loading an emoji-supporting font (adjust path if needed)
    try:
        font = ImageFont.truetype("seguiemj.ttf", 24)  # Segoe UI Emoji on Windows
    except:
        font = ImageFont.load_default()

    for i, row in enumerate(controls_text):
        if len(row) < 3:
            row += [""] * (3 - len(row))
        left, center, right = row
        y = 5 + i * 30
        draw.text((10, y), left, font=font, fill=(255, 255, 255))
        draw.text((col_width + 10, y), center, font=font, fill=(255, 255, 255))
        draw.text((2 * col_width + 10, y), right, font=font, fill=(255, 255, 255))

    # Convert to OpenCV format
    control_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Controls", control_image)
    cv2.moveWindow("Controls", 280, 40)


def clamp_pose_change(prev_pose, new_pose):
    clamped_pose = list(prev_pose)
    for i in range(3):
        delta = new_pose[i] - prev_pose[i]
        if abs(delta) > MAX_POS_DELTA:
            delta = np.sign(delta) * MAX_POS_DELTA
        clamped_pose[i] = prev_pose[i] + delta
    for i in range(3, 6):
        delta = new_pose[i] - prev_pose[i]
        if abs(delta) > MAX_ROT_DELTA:
            delta = np.sign(delta) * MAX_ROT_DELTA
        clamped_pose[i] = prev_pose[i] + delta
    if custom_x_limit:
        clamped_pose[0] = max(clamped_pose[0], custom_x_limit)
    if custom_y_limit:
        clamped_pose[1] = max(clamped_pose[1], custom_y_limit)
    if custom_z_limit:
        clamped_pose[2] = max(clamped_pose[2], custom_z_limit)
    return clamped_pose

def avg_hand_pos(landmark_list, shape):
    h, w, _ = shape
    keypoints = [0, 1, 2, 5, 9, 13, 17]
    xs = [landmark_list.landmark[i].x * w for i in keypoints]
    ys = [landmark_list.landmark[i].y * h for i in keypoints]
    return np.mean(xs), np.mean(ys)

def apply_tool_z_rotation(rotation_vector, delta_rz):
    current_rot = R.from_rotvec(rotation_vector)
    local_z_rot = R.from_euler('z', delta_rz)
    new_rot = current_rot * local_z_rot
    return new_rot.as_rotvec()

def draw_joint_angles(frame, joint_angles, warn_threshold=3.14, limit_threshold=5.75):
    for i, angle in enumerate(joint_angles):
        color = (0, 255, 0)
        if abs(angle) > warn_threshold:
            color = (0, 255, 255)
        if abs(angle) > limit_threshold:
            color = (0, 0, 255)
        text = f"J{i+1}: {angle:.2f} rad"
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

first = True

while True:
    show_control_window()

    updated_rotation = False
    ret_main, frame_main = cap_main.read()
    if cap_depth:
        ret_depth, frame_depth = cap_depth.read()
    else:
        ret_depth, frame_depth = False, None

    if not ret_main:
        break

    if cap_depth and not ret_depth:
        print("[WARNING] Depth camera disconnected or failed. Disabling depth camera processing. 2D motion only.")
        cap_depth = False
        frame_depth = None

    frame_main = cv2.flip(frame_main, 1)
    rgb_main = cv2.cvtColor(frame_main, cv2.COLOR_BGR2RGB)
    if cap_depth:
        frame_depth = cv2.flip(frame_depth, 1)
        rgb_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2RGB)

    if keyboard.is_pressed('space'):
        paused = not paused
        print("Paused" if paused else "Resumed")
        while keyboard.is_pressed('space'):
            pass

    if not paused:
        results_main = hands.process(rgb_main)
        results_depth = hands.process(rgb_depth) if cap_depth else None

    curr_pose = pose.copy()
    updated = False

    if results_main.multi_hand_landmarks and results_main.multi_handedness:
        rocknroll_detected = False
        down_rocknroll_detected = False
        for hand_landmarks, handedness in zip(results_main.multi_hand_landmarks, results_main.multi_handedness):
            label = handedness.classification[0].label
            mp_drawing.draw_landmarks(frame_main, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            g = gestures(landmarks, label)

            if g:
                if g[0] == 'rocknroll':
                    rocknroll_detected = True
                    if update_gesture_timer('rocknroll', True):
                        gripper.open_gripper(robot)
                        custom_z_limit = -0.2955
                        time.sleep(1)
                        robot.init_realtime_control()

                elif g[0] == 'down rocknroll':
                    down_rocknroll_detected = True
                    if update_gesture_timer('down rocknroll', True):
                        gripper.close_gripper(robot)
                        custom_z_limit = pose[2]
                        time.sleep(1)
                        robot.init_realtime_control()

                if not rocknroll_detected:
                    update_gesture_timer('rocknroll', False)
                if not down_rocknroll_detected:
                    update_gesture_timer('down rocknroll', False)
                write_gest = 50 if label == 'Right' else 80
                cv2.putText(frame_main, f'{(g[1]).title()}: {g[0]} detected!', (160, write_gest), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if not paused and label == 'Right':
                shape = frame_depth.shape if cap_depth else frame_main.shape
                curr_pos = avg_hand_pos(hand_landmarks, shape)
                gesture_targets = {
                    'point up': .0001,
                    'peace': .0003,
                    'three': .0005,
                    'open': .0007,
                }
                if prev_pos_main is not None and curr_pos is not None:
                    dx = curr_pos[0] - prev_pos_main[0]
                    dy = curr_pos[1] - prev_pos_main[1]
                    if g[0] in gesture_targets:
                        scale = gesture_targets[g[0]]
                        if cap_depth:
                            pose[0] += dx * scale
                            pose[2] -= dy * scale
                        else:
                            pose[1] -= dy * scale
                            pose[0] += dx * scale
                        updated = True
                prev_pos_main = curr_pos

    if cap_depth and results_depth and results_depth.multi_hand_landmarks and results_depth.multi_handedness:
        for hand_landmarks, handedness in zip(results_depth.multi_hand_landmarks, results_depth.multi_handedness):
            label = handedness.classification[0].label
            mp_drawing.draw_landmarks(frame_depth, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            g = gestures([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], label)
            if not paused and label == 'Right':
                curr_pos = avg_hand_pos(hand_landmarks, frame_main.shape)
                gesture_targets = {
                    'point up': .0001,
                    'peace': .0003,
                    'three': .0005,
                    'open': .0007,
                }
                if prev_pos_depth is not None and curr_pos is not None:
                    dx = curr_pos[0] - prev_pos_depth[0]
                    if g[0] == 'open':
                        pose[1] -= dx * scale
                        updated = True
                prev_pos_depth = curr_pos

    if not updated_rotation:
        pose[3:6] = curr_pose[3:6]

    if updated:
        try:
            pose[2] = max(pose[2], custom_z_limit)
            current_joints = robot.get_actual_joint_positions()
            if recovery_frames_remaining > 0:
                pose = clamp_pose_change(curr_pose, pose)
                robot.set_realtime_pose(pose)
                recovery_frames_remaining -= 1
            elif current_joints is not None:
                joint_within_limits = all(-6.0 <= angle <= 6.0 for angle in current_joints)
                if joint_within_limits:
                    pose = clamp_pose_change(curr_pose, pose)
                    robot.set_realtime_pose(pose)
                    joint_limit_blocked = False
                else:
                    print("[WARNING] Joint limit exceeded — motion blocked.")
                    joint_limit_blocked = True
                    target_joints = robot.get_inverse_kin(pose)
                    correction_ok = all(
                        (cj > 5.8 and tj < cj) or (cj < -5.8 and tj > cj) or (-5.8 <= tj <= 5.8)
                        for cj, tj in zip(current_joints, target_joints)
                    )
                    if correction_ok:
                        robot.init_realtime_control()
                        print("[RECOVERY] Joint correction motion allowed.")
                        robot.set_realtime_pose(pose)
                        recovery_frames_remaining = 10
                        joint_limit_blocked = False
            else:
                print("[ERROR] Failed to read joints — skipping motion.")
        except Exception as e:
            print(f"[EXCEPTION] during pose application: {e}")
    if first:
        first = False
        paused = True
    height = frame_main.shape[0]
    width_main = frame_main.shape[1]
    if paused:
        cv2.line(frame_main, (200, 60), (435, 60), (0, 0, 255), 50)
        cv2.putText(frame_main, "Motion Paused", (200, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    if cap_depth:
        height = min(frame_main.shape[0], frame_depth.shape[0])
        width_main = int(frame_main.shape[1] * (height / frame_main.shape[0]))
        width_depth = int(frame_depth.shape[1] * (height / frame_depth.shape[0]))
        resized_main = cv2.resize(frame_main, (width_main, height))
        resized_depth = cv2.resize(frame_depth, (width_depth, height))
        combined_frame = np.hstack((resized_main, resized_depth))
        left_roi = combined_frame[0:height, 0:width_main]
    else:
        combined_frame = frame_main.copy()
        left_roi = combined_frame

    current_joints = robot.get_actual_joint_positions()
    draw_joint_angles(left_roi, current_joints)

    if joint_limit_blocked:
        cv2.putText(combined_frame, "!! JOINT LIMIT BLOCKED !!", (160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Main (Left) | Depth (Right)", combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap_main.release()
if cap_depth:
    cap_depth.release()
cv2.destroyAllWindows()

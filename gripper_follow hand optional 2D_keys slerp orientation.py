import cv2
import mediapipe as mp
from URBasic import robotModel, urScriptExt
from Gestures import gestures
import numpy as np
import math
import time
import keyboard
from scipy.spatial.transform import Rotation as R, Slerp
import GripperFunctions as gripper
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import numpy as np

"""
[VERSION: Latest | Dual-Hand Active | Gesture-Gated Control | Orientation-Only Left Hand]

This version:
✅ Supports simultaneous left and right hand tracking
✅ Allows translation only when specific right-hand gestures are held:
   - One finger: slow
   - Peace: medium
   - Three: fast
   - Open: fastest
✅ Allows orientation control only with left-hand gestures (thumbs/points)
✅ Restricts tool Z-axis rotation to specific left-hand gestures (pinky up/down)
✅ Includes gripper control via left-hand rocknroll gestures
✅ Implements joint limit checking and automatic recovery
✅ Applies Z-limit clamping to avoid table collisions
✅ Motion smoothing via velocity clamping
✅ Debounces orientation control to avoid constant resetting
❌ Motion freezes if both hands are in frame with no valid gesture (safety-oriented)
❌ No quaternion interpolation (Slerp removed for simplicity)
❌ No dynamic gesture blending (only one gesture per hand is active at a time)

Summary:
→ The safest and most deliberate version so far — gesture-gated motion ensures
user intent, supports both hands at once, and prevents false movements.
Best for precise, stable control with low error risk.
"""

"""
This code uses a dual-camera setup (main + optional depth) to define robot motion in 2D or 3D based on hand gestures.

  - One finger:     Slowest speed
  - Two fingers:    Medium speed
  - Three fingers:  Fast
  - Open palm:      Fastest

  - Pinky up/down: Rotates the tool around its Z-axis
  - pinky with index up: opens tool (must be held for 0.25s)
  - middle and ring up / pinky with index down: closes tool (must be held for 0.25s)

Other features:
  - Real-time control
  - Joint safety limits and motion clamping
  - Pause/resume with space bar
  - Motion smoothing with velocity limits to reduce flinching

Notes:
  - Keep hands in a fist to prevent motion.
  - Be aware of background hands interfering with tracking.
  - when only one camera is plugged in, motion changes to x,y w thumbs up and down for z
"""

# === Robot setup ===
robot_ip = '192.168.1.10'
rtde_conf_file = 'URBasic/rtdeConfigurationDefault.xml'

model = robotModel.RobotModel()
robot = urScriptExt.UrScriptExt(host=robot_ip, robotModel=model, conf_filename=rtde_conf_file)

# === Reset robot to face-down position at start ===
start_pose = robot.get_actual_tcp_pose()
face_down_pose = [-0.03, 0.770, -0.170, 0.124, 3.14, 0.0]  # change orientation to be face down
print('=== Moving to Face Down ===')
robot.movej(robot.get_inverse_kin(face_down_pose), 0.5, 0.5)
pose = face_down_pose.copy()
robot.init_realtime_control()

cap_main = None
cap_depth = None

cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(0)

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

override_limit = False
shift_key_state = False

orientation_log = []

recovery_frames_remaining = 0
step = 0.003
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

MAX_POS_DELTA = 0.05
MAX_ROT_DELTA = 0.1

controls_text = [
    ["____ Controls: ____", "____ Speed Modes: ____","____ Single Camera (2D Motion): ____"],
    ["⌨Spacebar: Pause/Resume Shift: limit override", "👆First Finger(R): Slowest", "👍Thumbs Up (R): Go Up"],
    ["🤘Rocknroll Up/Down: Open/Close Gripper", "✌First & Second(R): Medium", "👎Thumbs Down (R): Go Down"],
    ["☝Pinky Up/Down: Tool Rotation", "     First-Third (R): Fast","Notes:  Use a ✊fist to reset hand"],
    ["⌨W,S,A,D,E,Q: Orientation ^,v,<,>,x,O", "🤚Open Palm(R): Fastest", "          for next movement"]
]

def show_control_window():
    width = 1200
    height = 31 * len(controls_text)
    col_width = width // 3

    # Create a blank image using PIL
    image = Image.new("RGB", (width, height), (50, 50, 175))
    draw = ImageDraw.Draw(image)

    # Try loading an emoji-supporting font (adjust path if needed)
    try:
        font = ImageFont.truetype("seguiemj.ttf", 18)  # Segoe UI Emoji on Windows
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
    cv2.moveWindow("Controls", 180, 40)


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
    if not override_limit:
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
    """Rotate current TCP orientation around tool Z-axis by delta_rz radians."""
    current_rot = R.from_rotvec(rotation_vector)
    local_z_rot = R.from_euler('z', delta_rz)
    new_rot = current_rot * local_z_rot  # Tool frame rotation
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

def get_orientation_quat(gesture):
    """
    Maps left-hand gestures to absolute target orientations (in Euler angles),
    which are converted to quaternions for smooth interpolation (slerp).
    """
    orientation_map = {
        'w': [0.0, 0.0, 0.0],
        's': [math.pi, 0.0, 0.0],
        'a': [math.pi / 2, 0.0, 0.0],
        'd': [-math.pi / 2, 0.0, 0.0],
        'e': [0.0, math.pi / 2, 0.0],
        'q': [0.0, -math.pi / 2, 0.0],
    }
    if gesture in orientation_map:
        return R.from_euler('xyz', orientation_map[gesture]).as_quat()
    return None

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
                if g[0] == 'pinky down':
                    pose[3:6] = apply_tool_z_rotation(pose[3:6], -0.05)
                    updated = True
                    updated_rotation = True
                elif g[0] == 'pinky':
                    pose[3:6] = apply_tool_z_rotation(pose[3:6], 0.05)
                    updated = True
                    updated_rotation = True

                elif g[0] == 'rocknroll':
                    rocknroll_detected = True
                    if update_gesture_timer('rocknroll', True):
                        gripper.open_gripper(robot)
                        custom_z_limit = -6.0
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

                cv2.putText(frame_main, f'{(g[1]).title()}: {g[0]} detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if not paused and label == 'Right':
                shape = frame_depth.shape if cap_depth else frame_main.shape
                curr_pos = avg_hand_pos(hand_landmarks, shape)
                gesture_targets = {
                    'point up': .00015,
                    'peace': .0005,
                    'three': .001,
                    'open': .00125,
                }
                if prev_pos_main is not None and curr_pos is not None:
                    dx = curr_pos[0] - prev_pos_main[0]
                    dy = curr_pos[1] - prev_pos_main[1]
                    if g[0] in gesture_targets:
                        scale = gesture_targets[g[0]]
                        if cap_depth:
                            pose[1] += dx * scale
                            pose[2] -= dy * scale
                        else:
                            pose[0] += dy * scale
                            pose[1] += dx * scale
                        updated = True
                    if not cap_depth:
                        z_step = 0.002
                        z_step = 0.001 if override_limit else step
                        if g[0] == 'thumbs up':
                            pose[2] += z_step
                            updated = True
                        elif g[0] == 'thumbs down':
                            pose[2] -= z_step
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
                    'point up': .00015,
                    'peace': .0005,
                    'three': .001,
                    'open': .00125,
                }
                if prev_pos_depth is not None and curr_pos is not None:
                    dx = curr_pos[0] - prev_pos_depth[0]
                    if g[0] in gesture_targets:
                        pose[0] += dx * scale
                        updated = True
                prev_pos_depth = curr_pos

    pressed_keys = [key for key in ['w', 's', 'a', 'd', 'q', 'e'] if keyboard.is_pressed(key)]

    if pressed_keys:
        # For simplicity, take the first pressed key to determine orientation
        key = pressed_keys[0]
        target_quat = get_orientation_quat(key)
        if target_quat is not None:
            curr_rot = R.from_rotvec(pose[3:6])
            curr_quat = curr_rot.as_quat()
            slerp = Slerp([0, 1], R.concatenate([R.from_quat(curr_quat), R.from_quat(target_quat)]))
            interp_rot = slerp(0.05)
            pose[3:6] = interp_rot.as_rotvec()
            updated = True
            updated_rotation = True

    if not updated_rotation:
        pose[3:6] = curr_pose[3:6]

    if updated_rotation:
        quat = R.from_rotvec(pose[3:6]).as_quat()
        timestamp = time.time()
        orientation_log.append((timestamp, *quat))

    # Z-limit override toggle
    if keyboard.is_pressed('shift'):
        if not shift_key_state:
            override_limit = not override_limit
            print("limit override:", override_limit)
            shift_key_state = True
    else:
        shift_key_state = False

    if updated:
        try:
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

    if override_limit:
        cv2.putText(combined_frame, "!! LIMIT OVERRIDE ENABLED !!", (80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    current_joints = robot.get_actual_joint_positions()
    draw_joint_angles(left_roi, current_joints)

    if joint_limit_blocked:
        cv2.putText(combined_frame, "!! JOINT LIMIT BLOCKED !!", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Main (Left) | Depth (Right)", combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap_main.release()
if cap_depth:
    cap_depth.release()

df = pd.DataFrame(orientation_log, columns=['timestamp', 'qx', 'qy', 'qz', 'qw'])
df.to_csv("slerp_orientation_quaternions.csv", index=False)
print("Saved orientation data to 'slerp_orientation_quaternions.csv'")

# Convert quaternions to orientation vectors (Z-axis direction)
orientation_vectors = [R.from_quat([q[1], q[2], q[3], q[4]]).apply([0, 0, 1]) for q in orientation_log]

# Normalize color gradient across the length of data
num_points = len(orientation_vectors)
colors = plt.cm.plasma(np.linspace(0, 1, num_points))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Origin
origin = np.array([0, 0, 0])

# Define a 2D circle in the XY plane
theta = np.linspace(0, 2 * np.pi, 100)
circle_radius = 0.3
circle_x = circle_radius * np.cos(theta)
circle_y = circle_radius * np.sin(theta)
circle_z = np.zeros_like(theta)  # Flat circle in XY plane (z = 0)

# Plot the circle
ax.plot(circle_x, circle_y, circle_z, color='gray', linewidth=1.0, linestyle='--')
ax.plot(circle_z, circle_x, circle_y, color='gray', linewidth=1.0, linestyle='--')
ax.plot(circle_y, circle_z, circle_x, color='gray', linewidth=1.0, linestyle='--')

# Draw only tip points of the orientation vectors
for i, vec in enumerate(orientation_vectors):
    end = origin + 0.3 * np.array(vec)  # scale for visibility
    ax.scatter(end[0], end[1], end[2], color=colors[i], s=12)  # s = size of point

# Labels and view
ax.set_title("Quaternion Orientation Tips with Reference Sphere")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-0.3, 0.3])
ax.set_ylim([-0.3, 0.3])
ax.set_zlim([-0.3, 0.3])
ax.view_init(elev=30, azim=135)  # Adjust for better viewing angle
plt.tight_layout()
plt.show()

cv2.destroyAllWindows()

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

"""
[VERSION: Early Prototype | Orientation-Rich | Risky Z | Left-Hand Gripper]

This version:
✅ Uses left-hand gestures for orientation (main & depth camera supported)
✅ Implements smooth orientation transitions via quaternion Slerp
✅ Includes basic joint limit checks and auto-recovery
✅ Supports gripper control via left-hand gestures
❌ Does NOT clamp Z-axis (can allow dangerous downward movement)
❌ Lacks gesture debounce for gripper
❌ Slightly inconsistent hand roles (gripper is left, motion is right)

Summary:
→ An older, flexible version with strong orientation support,
but lacks important safety constraints and gesture consistency.
"""

"""
This code uses a dual-camera setup (main + optional depth) to define robot motion in 2D or 3D based on hand gestures.

Right-hand gestures:
  - One finger:     Slowest speed
  - Two fingers:    Medium speed
  - Three fingers:  Fast
  - Open palm:      Fastest

Left-hand gestures:
  - Orientation control (thumbs/points): Sets target TCP orientation
  - Pinky up/down: Rotates the tool around its Z-axis

Other features:
  - Real-time control (non-blocking)
  - Joint safety limits and motion clamping
  - Pause/resume with spacebar
  - Motion smoothing with velocity limits to reduce flinching

Notes:
  - Keep hands in a fist to prevent motion.
  - Be aware of background hands interfering with tracking.
  - If you want all gestures on the same hand, change `label == 'Left'` to `'Right'` (twice).
"""

# === Robot setup ===
robot_ip = '192.168.1.10'  # Change to your robot IP
rtde_conf_file = 'URBasic/rtdeConfigurationDefault.xml'

# Real-time control robot instance
model = robotModel.RobotModel()
robot = urScriptExt.UrScriptExt(host=robot_ip, robotModel=model, conf_filename=rtde_conf_file)
robot.init_realtime_control()

pose = robot.get_actual_tcp_pose()

cap_main = None
cap_depth = None

# Try both camera indices to assign main (2D) and optional depth (X-axis motion) cameras
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

recovery_frames_remaining = 0  # Cooldown counter to allow recovery motion

orientation_motion = None
step = 0.005
scale = 0.7
paused = False
joint_limit_blocked = False  # Track if joint limits block motion

# === Velocity limiting parameters ===
# Max allowed motion per frame to prevent sudden robot jumps due to gesture misreads
MAX_POS_DELTA = 0.03   # meters per frame
MAX_ROT_DELTA = 0.1   # radians per frame

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

# Clamp X/Y/Z position deltas and RX/RY/RZ orientation deltas
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


def avg_hand_pos(landmarks, shape):
    """
    Calculate the average 2D position of the hand based on key stable palm landmarks.
    Used to determine the hand's movement across frames.
    """
    h, w, _ = shape
    keypoints = [0, 1, 2, 5, 9, 13, 17]  # More stable palm landmarks
    xs = [landmarks[i].x * w for i in keypoints]
    ys = [landmarks[i].y * h for i in keypoints]
    return np.mean(xs), np.mean(ys)

def apply_tool_z_rotation(rotation_vector, delta_rz):
    """Rotate current TCP orientation around tool Z-axis by delta_rz radians."""
    current_rot = R.from_rotvec(rotation_vector)
    local_z_rot = R.from_euler('z', delta_rz)
    new_rot = current_rot * local_z_rot  # Tool frame rotation
    return new_rot.as_rotvec()

def draw_joint_angles(frame, joint_angles, warn_threshold=3.14, limit_threshold=5.75):
    """
    Visual debug overlay: Shows joint angles with warnings if close to or exceeding limits.
    - Green: Safe
    - Yellow: Caution (approaching limit)
    - Red: Limit exceeded
    """
    for i, angle in enumerate(joint_angles):
        color = (0, 255, 0)  # Green
        if abs(angle) > warn_threshold:
            color = (0, 255, 255)  # Yellow
        if abs(angle) > limit_threshold:
            color = (0, 0, 255)  # Red

        text = f"J{i+1}: {angle:.2f} rad"
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def get_orientation_quat(gesture):
    """
    Maps left-hand gestures to absolute target orientations (in Euler angles),
    which are converted to quaternions for smooth interpolation (slerp).
    """
    orientation_map = {
        'thumbs up': [0.0, 0.0, 0.0],
        'thumbs down': [math.pi, 0.0, 0.0],
        'point left': [math.pi / 2, 0.0, 0.0],
        'point right': [-math.pi / 2, 0.0, 0.0],
        'forward': [0.0, math.pi / 2, 0.0],
        'back': [0.0, -math.pi / 2, 0.0],
    }
    if gesture in orientation_map:
        return R.from_euler('xyz', orientation_map[gesture]).as_quat()
    return None

while True:
    main_orientation_active = False
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

    # Pause toggle
    if keyboard.is_pressed('space'):
        paused = not paused
        print("Paused" if paused else "Resumed")
        while keyboard.is_pressed('space'):
            pass

    if not paused:
        results_main = hands.process(rgb_main)
        if cap_depth:
            results_depth = hands.process(rgb_depth)
        else:
            results_depth = None
    curr_pose = pose.copy()
    updated = False

    if results_main.multi_hand_landmarks and results_main.multi_handedness:
        for hand_landmarks, handedness in zip(results_main.multi_hand_landmarks, results_main.multi_handedness):
            label = handedness.classification[0].label
            mp_drawing.draw_landmarks(frame_main, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            g = gestures(landmarks, label)

            if g and label == 'Left':
                # Orientation gestures
                target_quat = get_orientation_quat(g[0])
                if target_quat is not None:
                    # Interpolate to orientation
                    curr_rot = R.from_rotvec(pose[3:6])
                    curr_quat = curr_rot.as_quat()
                    slerp = Slerp([0, 1], R.concatenate([R.from_quat(curr_quat), R.from_quat(target_quat)]))
                    interp_rot = slerp(0.05)
                    new_rotvec = interp_rot.as_rotvec()
                    pose[3:6] = new_rotvec
                    updated = True
                    main_orientation_active = True

                # Wrist gestures
                elif g[0] == 'pinky down':
                    pose[3:6] = apply_tool_z_rotation(pose[3:6], 0.05)
                    updated = True
                elif g[0] == 'pinky':
                    pose[3:6] = apply_tool_z_rotation(pose[3:6], -0.05)
                    updated = True
            if g and label == 'Right':
                if g[0] == 'rocknroll':
                    gripper.open_gripper(robot)
                    time.sleep(1)
                    robot.init_realtime_control()
                elif g[0] == 'down rocknroll':
                    gripper.close_gripper(robot)
                    time.sleep(1)
                    robot.init_realtime_control()


                cv2.putText(frame_main, f'{(g[1]).title()}: {g[0]} detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)

            if not paused and label == 'Right':
                if cap_depth:
                    curr_pos = avg_hand_pos(hand_landmarks.landmark, frame_depth.shape)
                else:
                    curr_pos = avg_hand_pos(hand_landmarks.landmark, frame_main.shape)
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
                        pose[1] += dx * scale
                        pose[2] -= dy * scale
                        updated = True
                prev_pos_main = curr_pos

    if cap_depth:
        if results_depth.multi_hand_landmarks and results_depth.multi_handedness:
            for hand_landmarks, handedness in zip(results_depth.multi_hand_landmarks, results_depth.multi_handedness):
                label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame_depth, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                g = gestures([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], label)

                if g and label == 'Left' and not main_orientation_active:
                    if g[0] == 'point left':
                        g = 'back'
                    elif g[0] == 'point right':
                        g = 'forward'
                    target_quat = get_orientation_quat(g)
                    if target_quat is not None:
                        curr_rot = R.from_rotvec(pose[3:6])
                        curr_quat = curr_rot.as_quat()
                        slerp = Slerp([0, 1], R.concatenate([R.from_quat(curr_quat), R.from_quat(target_quat)]))
                        interp_rot = slerp(0.05)
                        pose[3:6] = interp_rot.as_rotvec()
                        updated = True
                        cv2.putText(frame_depth, f'{label}: {g} detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)

                if not paused and label == 'Right':
                    curr_pos = avg_hand_pos(hand_landmarks.landmark, frame_main.shape)
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

    # === Apply clamped pose if updated and within joint limits ===
    if updated:
        try:
            current_joints = robot.get_actual_joint_positions()

            if recovery_frames_remaining > 0:
                # We're in a recovery window — skip joint limit checks
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
                        recovery_frames_remaining = 10  # allow recovery motion for ~10 frames
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

    current_joints = robot.get_actual_joint_positions()
    draw_joint_angles(left_roi, current_joints)

    # Show joint limit warning if blocked
    if joint_limit_blocked:
        cv2.putText(combined_frame, "!! JOINT LIMIT BLOCKED !!", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Main (Left) | Depth (Right)", combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up cameras and windows
cap_main.release()
if cap_depth:
    cap_depth.release()
cv2.destroyAllWindows()
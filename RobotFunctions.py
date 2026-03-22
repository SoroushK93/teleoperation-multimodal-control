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


# ---determines which cameras are open and arranges them accordingly---

cap_main = None
cap_depth = None

def get_num_cameras(cap0, cap1):
    if cap1.isOpened():
        cap_main = cap1
        cap_depth = cap0 if cap0.isOpened() else False
    elif cap0.isOpened():
        cap_main = cap0
        cap_depth = False
    else:
        raise RuntimeError("No usable webcam found.")
    return cap_main, cap_depth


# --- creates instructions for use above the camera ---

controls_text = [
    ["____ Controls: ____", "____ Speed Modes (right hand): ____","____ Single Camera (2D Motion): ____"],
    ["⌨Spacebar: Pause/Resume", "👆First Finger: Slowest", "👍Thumbs Up (R): Go Up"],
    ["🤘Rocknroll: Open Gripper", "✌First & Second: Medium", "👎Thumbs Down (R): Go Down"],
    ["     Rocknroll Down: Close Gripper", "     First-Third: Fast","Notes: Use a ✊fist to reset hand"],
    ["☝Pinky Up/Down: Tool Rotation", "🤚Open Palm: Fastest", "           for next movement"]
]

control_window_name = "Controls"

def show_control_window(controls_text):
    width = 1000
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
    cv2.moveWindow("Controls", 280, 40)


# --- clamps speed of robot as well as min and max locations ---

# ---custom limits---
# x

# y
base = 0.225
# z
rolling_table = -0.2955
utility_box = -0.653

def clamp_pose_change(prev_pose, new_pose, max_pos_delta=.05, max_rot_delta=.1, custom_x_limit=None, custom_y_limit=None, custom_z_limit=None):
    clamped_pose = list(prev_pose)
    for i in range(3):
        delta = new_pose[i] - prev_pose[i]
        if abs(delta) > max_pos_delta:
            delta = np.sign(delta) * max_pos_delta
        clamped_pose[i] = prev_pose[i] + delta
    for i in range(3, 6):
        delta = new_pose[i] - prev_pose[i]
        if abs(delta) > max_rot_delta:
            delta = np.sign(delta) * max_rot_delta
        clamped_pose[i] = prev_pose[i] + delta
    if custom_x_limit:
        clamped_pose[0] = max(clamped_pose[0], custom_x_limit)
    if custom_y_limit:
        clamped_pose[1] = max(clamped_pose[1], custom_y_limit)
    if custom_z_limit:
        clamped_pose[2] = max(clamped_pose[2], custom_z_limit)
    return clamped_pose


# --- averages palm landmarks ---
def avg_hand_pos(landmark_list, shape):
    h, w, _ = shape
    keypoints = [0, 1, 2, 5, 9, 13, 17]
    xs = [landmark_list.landmark[i].x * w for i in keypoints]
    ys = [landmark_list.landmark[i].y * h for i in keypoints]
    return np.mean(xs), np.mean(ys)


# --- moves coordinate system to last joint and only rotates it in the z dir ---
def apply_tool_z_rotation(rotation_vector, delta_rz):
    current_rot = R.from_rotvec(rotation_vector)
    local_z_rot = R.from_euler('z', delta_rz)
    new_rot = current_rot * local_z_rot
    return new_rot.as_rotvec()


# --- writes current joint locations on top left of camera ---
def draw_joint_angles(frame, joint_angles, warn_threshold=3.14, limit_threshold=5.75):
    for i, angle in enumerate(joint_angles):
        color = (0, 255, 0)
        if abs(angle) > warn_threshold:
            color = (0, 255, 255)
        if abs(angle) > limit_threshold:
            color = (0, 0, 255)
        text = f"J{i+1}: {angle:.2f} rad"
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# --- establishes direction for slerp function ---
def get_orientation_quat(gesture):
    orientation_map = {
        'w': [0.0, 0.0, 0.0],
        's': [math.pi, 0.0, 0.0],
        'a': [math.pi / 2, 0.0, 0.0],
        'd': [-math.pi / 2, 0.0, 0.0],
        'q': [0.0, math.pi / 2, 0.0],
        'e': [0.0, -math.pi / 2, 0.0],
    }
    if gesture in orientation_map:
        return R.from_euler('xyz', orientation_map[gesture]).as_quat()
    return None


# --- creates a need to hold the gesture before there is a reaction ---

gesture_timers = {
    'rocknroll': 0.0,
    'down rocknroll': 0.0
}

def update_gesture_timer(gesture_name, is_detected, gesture_hold_threshold=0.33, gesture_timers=gesture_timers):
    current_time = time.time()
    if is_detected:
        if gesture_timers[gesture_name] == 0.0:
            gesture_timers[gesture_name] = current_time
        elif current_time - gesture_timers[gesture_name] > gesture_hold_threshold:
            gesture_timers[gesture_name] = 0.0
            return True
    else:
        gesture_timers[gesture_name] = 0.0
    return False


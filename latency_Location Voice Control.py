import openai
import json
import os
import time
import numpy as np
# Removed unused vision imports
# import cv2
# import pyrealsense2 as rs
# import cv2.aruco as aruco
from URBasic import robotModel, urScriptExt
import threading
import speech_recognition as sr
import pyaudio
import tkinter as tk
from tkinter import ttk
import GripperFunctions
import pandas as pd
import os


# =================================================================================
# PART 1: CONFIGURATION & INITIALIZATION
# =================================================================================

# --- OpenAI Configuration ---
try:
    # It's recommended to set the API key as an environment variable
    if "OPENAI_API_KEY" not in os.environ:
        # Replace with your key if you must hardcode, but environment variable is safer
        os.environ["OPENAI_API_KEY"] = "..."

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "YOUR_API_KEY_HERE":
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Please set the OPENAI_API_KEY environment variable or replace the placeholder in the script.")
        exit()

    client = openai.OpenAI()
    LLM_MODEL = "gpt-4o" # Ensure this model is capable of JSON output
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit()

# --- Robot Configuration ---
ROBOT_IP = '192.168.1.10'
RTDE_CONF_FILE = 'URBasic/rtdeConfigurationDefault.xml'
TCP_ORIENTATION = np.array([0.0, 3.14, 0.0], dtype=np.float32)

home_joints = np.array(
    [np.radians(93.52), np.radians(-150.91), np.radians(-81.22), np.radians(-37.91), np.radians(89.67),
     np.radians(183.09)])

# --- Safety Configuration ---
SAFE_ZONE_X = [-0.60, 0.575]  # Min and Max X in meters
SAFE_ZONE_Y = [-0.60, 1.200]  # Min and Max Y in meters
SAFE_ZONE_Z = [-0.3, 1.0]  # Min and Max Z in meters

# --- Global Variables ---
keep_running = True
latency_records = []

# =================================================================================
# PART 3: ROBOT ACTION FUNCTIONS (with Safety Checks)
# =================================================================================

def is_pose_safe(pose):
    """Checks if a target pose is within the defined SAFE_ZONE."""
    x, y, z = pose[0], pose[1], pose[2]
    if not (SAFE_ZONE_X[0] <= x <= SAFE_ZONE_X[1]):
        return False, f"X-coordinate {x:.3f} is outside the safe zone {SAFE_ZONE_X}"
    if not (SAFE_ZONE_Y[0] <= y <= SAFE_ZONE_Y[1]):
        return False, f"Y-coordinate {y:.3f} is outside the safe zone {SAFE_ZONE_Y}"
    if not (SAFE_ZONE_Z[0] <= z <= SAFE_ZONE_Z[1]):
        return False, f"Z-coordinate {z:.3f} is outside the safe zone {SAFE_ZONE_Z}"
    return True, "Pose is within safe limits."

def safe_move_to_pose(robot, pose, speed=0.1, acceleration=0.25):
    """Checks pose safety before moving the robot."""
    is_safe, message = is_pose_safe(pose)
    if is_safe:
        robot.movel(pose=pose, a=acceleration, v=speed)
        return f"Move executed to {np.round(pose[:3], 3)}."
    else:
        error_message = f"Error: Safety violation. {message}. Move cancelled."
        print(error_message)
        return error_message

def move_home(robot):
    """Moves the robot to its home position."""
    print("Moving robot to home position...")
    robot.movej(q=home_joints, a=0.5, v=0.5)
    time.sleep(1) # Give robot time to settle
    return "Move home command executed."


def move_linear_pose(robot, x: float, y: float, z: float, rx: float = 0.0, ry: float = 3.14, rz: float = 0.0, speed: float = 0.1, acceleration: float = 0.25):
    """Moves the robot's tool in a straight line to a target Cartesian pose."""
    target_pose = [x, y, z, rx, ry, rz]
    return safe_move_to_pose(robot, pose=target_pose, speed=speed, acceleration=acceleration)


def move_relative_pose(robot, x: float = 0.0, y: float = 0.0, z: float = 0.0, speed: float = 0.1, acceleration: float = 0.25):
    """Moves the robot's tool by a specified offset from its current position."""
    current_pose = robot.get_actual_tcp_pose()
    if current_pose is None: return "Error: Could not get current robot pose."

    target_pose = [current_pose[0] + x, current_pose[1] + y, current_pose[2] + z,
                   current_pose[3], current_pose[4], current_pose[5]]
    return safe_move_to_pose(robot, pose=target_pose, speed=speed, acceleration=acceleration)

def open_gripper(robot):
    """Opens the gripper."""
    print("Opening gripper...")
    GripperFunctions.open_gripper(robot)
    return "Gripper opened."


def close_gripper(robot):
    """Closes the gripper."""
    print("Closing gripper...")
    GripperFunctions.close_gripper(robot)
    return "Gripper closed."


def get_current_pose(robot):
    """Gets the current Cartesian pose of the robot's tool."""
    pose = robot.get_actual_tcp_pose()
    if pose is None:
        return "Error: Could not get current robot pose."
    return f"Current pose: {np.round(pose, 3).tolist()}" # Return as list for cleaner output


# =================================================================================
# PART 4: LLM INTERACTION & GUI
# =================================================================================

SYSTEM_PROMPT = """
You are a Universal Robots UR10e controller. Your goal is to translate user commands into a sequence of function calls.
You MUST respond ONLY with a single JSON object containing "commands", which is a list of objects. Each object in the list MUST have "function_name" and "args" keys. Do NOT include any explanations outside the JSON.

**COORDINATE SYSTEM & UNITS (Standard Cartesian Interpretation):**
- **X-Axis:**
    - "FORWARD" means moving along the positive X axis.
    - "BACK" means moving along the negative X axis.
- Y-Axis:
    - "RIGHT" means POSITIVE Y.
    - "LEFT" means NEGATIVE Y.
- Z-Axis:
    - "UP" means POSITIVE Z.
    - "DOWN" means NEGATIVE Z.
- All linear units for movement and position MUST be in Meters.

**IMPORTANT:** This robot does NOT have a vision system. It CANNOT see or identify objects.
Therefore, you CANNOT plan tasks like "pick up the block", "sort shapes", or "find anything".
You can only execute explicit movements and gripper commands.

**AVAILABLE FUNCTIONS:**
- `move_home()`: Moves the robot to its home position.
- `move_linear_pose(x: float, y: float, z: float, rx: float = 0.0, ry: float = 3.14, rz: float = 0.0, speed: float = 0.1, acceleration: float = 0.25)`: Moves to an absolute Cartesian pose (x, y, z, rx, ry, rz). The rotational components (rx, ry, rz) default to [0.0, 3.14, 0.0] which is downward.
- `move_relative_pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, speed: float = 0.1, acceleration: float = 0.25)`: Moves the robot's tool by a specified offset from its current position. All arguments are optional and default to 0.0.
- `get_current_pose()`: Returns the current Cartesian pose of the robot's tool.
- `open_gripper()`: Opens the gripper.
- `close_gripper()`: Closes the gripper.

**Example Conversation Flow:**
User: "Move the robot home."
LLM: {"commands": [{"function_name": "move_home", "args": {}}]}

User: "Move forward by 10 centimeters."
LLM: {"commands": [{"function_name": "move_relative_pose", "args": {"x": 0.1}}]}

User: "Move back by 5 centimeters."
LLM: {"commands": [{"function_name": "move_relative_pose", "args": {"x": -0.05}}]}

User: "Move right 2 inches and then move up 1 inch."
LLM: {"commands": [{"function_name": "move_relative_pose", "args": {"y": 0.0508}}, {"function_name": "move_relative_pose", "args": {"z": 0.0254}}]}

User: "What is the current position?"
LLM: {"commands": [{"function_name": "get_current_pose", "args": {}}]}

**USER REQUEST BELOW**
"""


def get_llm_response(conversation_history):
    """Sends the user command to OpenAI and gets a JSON response."""
    # Ensure the system prompt is always the first message
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            response_format={"type": "json_object"}, # Explicitly request JSON
            temperature=0.0 # Keep temperature low for deterministic function calls
        )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "Error communicating with OpenAI: {e}"}}'


class RecorderApp:
    """A persistent GUI control panel for the robot."""

    def __init__(self, root, recognizer, command_dispatcher):
        self.root = root
        self.recognizer = recognizer
        self.command_dispatcher = command_dispatcher
        self.is_recording = False
        self.task_in_progress = False
        self.conversation_history = [] # Initialize conversation history here
        self._setup_gui()

        self.start_time = 0
        self.transcription_end_time = 0
        self.execution_end_time = 0
        self.total_llm_time = 0

    def _setup_gui(self):
        self.root.title("Robot Voice Controller")
        self.root.geometry("450x250")
        self.root.attributes('-topmost', True) # Keep window on top
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=10)
        self.status_label = ttk.Label(self.root, text="Press and hold the button to record.",
                                      font=("Helvetica", 10), wraplength=430)
        self.status_label.pack(pady=10)
        self.record_button = ttk.Button(self.root, text="Hold to Record")
        self.record_button.pack(pady=20, ipadx=20, ipady=10)
        self.exit_button = ttk.Button(self.root, text="Exit Application", command=self.exit_app)
        self.exit_button.pack(pady=10)
        self.record_button.bind("<ButtonPress-1>", self.start_recording)
        self.record_button.bind("<ButtonRelease-1>", self.stop_recording)

    def start_recording(self, event):
        if self.task_in_progress:
            self.status_label.config(text="Cannot record, a task is already in progress.")
            return
        if self.is_recording: return
        self.start_time = time.time()
        self.status_label.config(text="🔴 Recording...")
        self.record_button.config(text="Recording...")
        self.is_recording = True
        threading.Thread(target=self._record_and_process_audio, daemon=True).start()

    def stop_recording(self, event):
        if not self.is_recording: return
        self.is_recording = False
        self.status_label.config(text="Transcribing...")
        self.record_button.config(text="Hold to Record")

    def _record_and_process_audio(self):
        """Handles the entire audio lifecycle in one thread to prevent race conditions."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []
        while self.is_recording:
            try:
                frames.append(stream.read(1024))
            except IOError: # Catch potential PyAudio errors if recording stops abruptly
                print("PyAudio IOError during recording. Stream might have closed.")
                break
        stream.stop_stream()
        stream.close()
        p.terminate()

        if not frames:
            self.status_label.config(text="No audio recorded. Try again.")
            self.task_in_progress = False # Reset task_in_progress if no audio
            return

        audio_data = sr.AudioData(b''.join(frames), 16000, p.get_sample_size(pyaudio.paInt16))
        try:
            self.task_in_progress = True

            text = self.recognizer.recognize_openai(audio_data, model="whisper-1")
            self.transcription_end_time = time.time()
            self.status_label.config(text=f"Transcribed: {text}")
            print(f"\n> User Command: '{text}'") # Print immediately after transcription

            if any(word in text.lower().strip() for word in ['exit', 'stop', 'quit', 'shut down', 'terminate']):
                self.exit_app()
                return

            self._execute_task(text) # Call execute_task with the transcribed text

        except sr.UnknownValueError:
            self.status_label.config(text="Could not understand audio. Please try again.")
            self.task_in_progress = False
        except sr.RequestError as e:
            self.status_label.config(text=f"Could not request results from audio service; {e}")
            self.task_in_progress = False
        except Exception as e:
            self.status_label.config(text=f"An unexpected error occurred during audio processing: {e}")
            print(f"Error in _record_and_process_audio: {e}")
            self.task_in_progress = False

    def _report_timings(self):
        """Calculates and prints the timings for each stage."""
        transcription_time = self.transcription_end_time - self.start_time
        llm_time = self.total_llm_time
        total_time = self.execution_end_time - self.start_time
        execution_time = total_time - transcription_time - llm_time
        latency_records.append((transcription_time, llm_time, total_time, execution_time))

        print("\n--- Task Timing Report ---")
        print(f"Transcription Time: {transcription_time:.2f} seconds")
        print(f"LLM Processing Time (Total): {llm_time:.2f} seconds")
        print(f"Task Execution Time: {execution_time:.2f} seconds")
        print(f"Total Time: {total_time:.2f} seconds")
        print("--------------------------\n")

    def _execute_task(self, user_input):
        """Contains the planning and execution loop for a given user command."""
        self.status_label.config(text=f"Sending to AI: '{user_input}'")
        conversation_history = [{"role": "user", "content": user_input}]
        self.total_llm_time = 0
        self.conversation_history = [{"role": "user", "content": user_input}]

        while True:
            if not keep_running:
                print("Application shutting down, stopping task execution.")
                break
            llm_start_time = time.time()
            llm_json_str = get_llm_response(conversation_history)
            llm_end_time = time.time()
            self.total_llm_time += (llm_end_time - llm_start_time)

            llm_json_str = get_llm_response(self.conversation_history)
            print(f"LLM Response: {llm_json_str}")

            try:
                command_data = json.loads(llm_json_str)
            except json.JSONDecodeError:
                self.status_label.config(text=f"Error: AI returned invalid JSON: {llm_json_str}. Task aborted.")
                print(f"Error: AI returned invalid JSON: {llm_json_str}")
                break

            if "error" in command_data:
                self.status_label.config(text=f"AI Error: {command_data['error']}. Task aborted.")
                print(f"AI Error: {command_data['error']}")
                break

            commands_to_execute = command_data.get("commands", [])

            if not commands_to_execute:
                self.status_label.config(text="✅ Task Complete! Ready for new command.")
                self.execution_end_time = time.time()
                self._report_timings()
                self.status_label.config(text="✅ Task Complete! Ready for new command.")
                print("Task Complete!")
                break

            self.conversation_history.append({"role": "assistant", "content": llm_json_str})

            all_results = []
            error_occurred = False
            for command in commands_to_execute:
                function_name = command.get("function_name")
                args = command.get("args", {})

                self.status_label.config(text=f"Running: {function_name}({args})")
                print(f"  Executing: {function_name}({args})")

                # *** FIX APPLIED HERE: Changed lambda args to lambda **kw ***
                result = self.command_dispatcher.get(function_name,
                                                     lambda **kw: f"Error: Unknown function '{function_name}'")(
                                                         **args)
                print(f"  Result: {result}")
                all_results.append(str(result))
                if "Error" in str(result):
                    error_occurred = True
                    break

            context_for_llm = f"Function execution results: [{', '.join(all_results)}]. Based on these results, what is the next sequence of commands? If the task is finished, return an empty 'commands' list."
            self.conversation_history.append({"role": "user", "content": context_for_llm})

            if error_occurred:
                self.status_label.config(text=f"Error during execution. Please try again.")
                print("Error occurred during execution. Breaking task loop.")
                break

        self.task_in_progress = False


    def exit_app(self):
        """Gracefully shuts down the application."""
        global keep_running
        if keep_running:
            print("Exit button clicked. Shutting down.")
            keep_running = False
            if latency_records:  # Only save if there's something to write
                print("Saving all latency data to Excel...")

                latency_df = pd.DataFrame(latency_records,
                                          columns=['transcription time', 'llm time', 'total time', 'execution time'])

                folder_path = r"C:\Users\Lydia\Undergraduate Research w Dr. Korivand\trials\excel sheets from trials"
                os.makedirs(folder_path, exist_ok=True)

                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(folder_path, f"LLM Latency_{timestamp}.xlsx")

                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    latency_df.to_excel(writer, sheet_name='Latency Summary', index=False)

                print(f"✅ Saved latency log to: {filename}")
            self.root.quit()

    def run(self):
        """Starts the GUI event loop."""
        self.root.mainloop()


# =================================================================================
# Main Execution Block
# =================================================================================
def main():
    global keep_running
    robot = None

    try:
        print("Initializing Robot...")
        model = robotModel.RobotModel()
        # Corrected: Removed non-standard URBasic commands.
        # This is the correct and only necessary initialization from your original code.
        robot = urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=model, conf_filename=RTDE_CONF_FILE)
        print("Robot initialized and connected.")

        # Perform initial move home after connection
        move_home(robot)

        COMMAND_DISPATCHER = {
            "move_home": lambda **kw: move_home(robot),
            "move_linear_pose": lambda **kw: move_linear_pose(robot, **kw),
            "move_relative_pose": lambda **kw: move_relative_pose(robot, **kw),
            "get_current_pose": lambda **kw: get_current_pose(robot),
            "open_gripper": lambda **kw: open_gripper(robot),
            "close_gripper": lambda **kw: close_gripper(robot),
        }

        recognizer = sr.Recognizer()
        print("\nSetup complete. Launching command GUI...")

        root = tk.Tk()
        app = RecorderApp(root, recognizer, COMMAND_DISPATCHER)
        app.run()

    except SystemExit as e:
        print(f"System Exit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
    finally:
        print("\nShutting down all systems...")
        keep_running = False
        if robot:
            print("Closing robot connection...")
            robot.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
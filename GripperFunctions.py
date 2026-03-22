import URBasic
import time

# --- Configuration ---
ROBOT_IP = '192.168.1.10'
GRIPPER_SOCKET_HOST = '127.0.0.1'
GRIPPER_SOCKET_PORT = 63352
GRIPPER_SOCKET_NAME = 'gripper_socket'


# --- Low-Level URScript Generation Functions ---

def get_connect_script():
    """Returns the URScript to open the socket connection."""
    return f'''
  textmsg("Opening gripper socket")
  socket_open("{GRIPPER_SOCKET_HOST}", {GRIPPER_SOCKET_PORT}, "{GRIPPER_SOCKET_NAME}")
  sleep(1.0)
'''

def get_disconnect_script():
    """Returns the URScript to close the socket connection."""
    return f'''
  textmsg("Closing gripper socket")
  socket_close("{GRIPPER_SOCKET_NAME}")
'''

def get_activate_script():
    """Returns the URScript for the gripper activation sequence."""
    # This sequence is critical and requires specific pauses.
    return f'''
  socket_send_line("SET ACT 0", "{GRIPPER_SOCKET_NAME}")
  sleep(0.1)
  textmsg("Activating Gripper")
  socket_send_line("SET ACT 1", "{GRIPPER_SOCKET_NAME}")
  sleep(1.5)
  socket_send_line("SET GTO 1", "{GRIPPER_SOCKET_NAME}")
  sleep(0.5)
'''

def get_close_script():
    """Returns the URScript to close the gripper."""
    # CORRECTED: Changed sleep from 0.0 to 2.0 to allow time for motion.
    return f'''
  textmsg("Closing Gripper")
  socket_send_line("SET POS 255 255 255", "{GRIPPER_SOCKET_NAME}")
'''

def get_open_script():
    """Returns the URScript to open the gripper."""
    return f'''
  textmsg("Opening Gripper")
  socket_send_line("SET POS 0 255 255", "{GRIPPER_SOCKET_NAME}")
'''

def get_custom_gripper_script(position=0, speed=255, force=255):
    """
    Returns a URScript command to move the gripper to a specified position with given speed and force.
    """
    position = int(max(0, min(position, 255)))
    speed = int(max(0, min(speed, 255)))
    force = int(max(0, min(force, 255)))
    return f'''
  textmsg("Setting Gripper POS:{position} SPD:{speed} FRC:{force}")
  socket_send_line("SET POS {position} {speed} {force}", "{GRIPPER_SOCKET_NAME}")
  sleep(0.5)
'''

# --- High-Level Callable Action Functions ---

def activate_gripper(robot_conn):
    """
    Assembles and sends a complete script to ONLY activate the gripper.
    """
    print("--- Sending command: ACTIVATE GRIPPER ---") # Corrected print statement
    script = (
            get_connect_script() +
            get_activate_script() +
            get_disconnect_script()
    )
    robot_conn.robotConnector.RealTimeClient.SendProgram(script)
    robot_conn.waitRobotIdleOrStopFlag()
    print("--- ACTIVATE GRIPPER command complete ---") # Corrected print statement

def open_gripper(robot_conn):
    """
    Assembles and sends a complete script to open the gripper.
    """
    print("--- Sending command: OPEN GRIPPER ---")
    # RE-ENABLED: get_activate_script() is included for robust, standalone operation.
    script = (
        get_connect_script() +
        get_open_script() +
        get_disconnect_script()
    )
    robot_conn.robotConnector.RealTimeClient.SendProgram(script)
    robot_conn.waitRobotIdleOrStopFlag()
    print("--- OPEN GRIPPER command complete ---")

def close_gripper(robot_conn):
    """
    Assembles and sends a complete script to close the gripper.
    """
    print("--- Sending command: CLOSE GRIPPER ---")
    # RE-ENABLED: get_activate_script() is included for robust, standalone operation.
    script = (
        get_connect_script() +
        get_close_script() +
        get_disconnect_script()
    )
    robot_conn.robotConnector.RealTimeClient.SendProgram(script)
    robot_conn.waitRobotIdleOrStopFlag()
    print("--- CLOSE GRIPPER command complete ---")

def set_gripper(robot_conn, width_mm=52.0, speed=255, force=255):
    """
    Sets the gripper to an arbitrary width in mm, with specific speed and force.

    :param robot_conn: The robot connection object
    :param width_mm: Desired gripper opening width in mm (0 mm = fully closed, 52 mm = fully open)
    :param speed: Gripper speed (0–255)
    :param force: Gripper force (0–255)
    """
    width_mm = max(0.0, min(width_mm, 52.0))  # Clamp to valid range
    position = int(round(255 * (1 - (width_mm / 52.0))))
    speed = int(max(0, min(speed, 255)))
    force = int(max(0, min(force, 255)))

    print(f"--- Sending command: SET GRIPPER to {width_mm:.1f} mm (POS={position}, SPD={speed}, FRC={force}) ---")

    script = (
        get_connect_script() +
        get_custom_gripper_script(position, speed, force) +
        get_disconnect_script()
    )
    robot_conn.robotConnector.RealTimeClient.SendProgram(script)
    robot_conn.waitRobotIdleOrStopFlag()

    print("--- SET GRIPPER command complete ---")


def main():
    """
    Main function demonstrating how to call the individual functions.
    """
    print("--- Starting Callable Gripper Control Demo ---")

    robot = None
    try:
        # Initialize and connect to the robot
        print("Initializing robot model...")
        robot_model = URBasic.robotModel.RobotModel()
        print(f"Connecting to robot at {ROBOT_IP}...")
        robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robot_model)
        print("Resetting potential robot errors...")
        robot.reset_error()
        time.sleep(1)

        # --- Example of calling individual gripper actions ---
        print("\nExecuting individual gripper commands...")

        # First, just activate the gripper to ensure it's ready.
        activate_gripper(robot)
        time.sleep(1)

        # # Now, call the standalone close command.
        # close_gripper(robot)
        #
        # # Set gripper to 31mm width with medium speed and low force
        # set_gripper(robot, width_mm=34, speed=100, force=100)
        #
        # # Finally, call the standalone open command.
        # open_gripper(robot)

        print("\n--- Demo sequence finished. ---")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        if robot:
            try:
                robot.stopj(2)
            except Exception as stop_e:
                print(f"Could not send stop command during error handling: {stop_e}")
    finally:
        # Cleanly disconnect from the robot
        if robot:
            print("Closing connection to the robot.")
            robot.close()
        print("--- Script Finished ---")


if __name__ == '__main__':
    main()
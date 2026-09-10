import cv2
import threading
import time
import math
import numpy as np
import torch
import torch.nn as nn
import serial
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv

# ==========================================
# 0. Hardware & Odometry Parameters
# ==========================================
WHEEL_RADIUS_CM = 3.3       
TRACK_WIDTH_CM = 17.6       

robot_x = 0.0
robot_y = 0.0
robot_theta = 0.0 
x_path = [0.0]
y_path = [0.0]
first_reading = True
prev_angle_l = 0.0
prev_angle_r = 0.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_aug.pt")
BRAIN_PATH = os.path.join(SCRIPT_DIR, "robot_brain_best.pth")

# ==========================================
# 1. PyTorch DQN Architecture
# ==========================================
class DQN(nn.Module):
    def __init__(self, action_dim=3):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (80, 60))
    normalized = resized.astype(np.float32) / 255.0
    return torch.tensor(normalized).unsqueeze(0).unsqueeze(0)

# ==========================================
# 2. Local YOLO AI Setup
# ==========================================
try:
    local_model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

LEFT_CLASS_ID = 1   
RIGHT_CLASS_ID = 0  

custom_colors = [None, None]
custom_colors[LEFT_CLASS_ID] = sv.Color(r=255, g=220, b=0)     
custom_colors[RIGHT_CLASS_ID] = sv.Color(r=255, g=255, b=255)  
custom_palette = sv.ColorPalette(colors=custom_colors)

mask_annotator = sv.MaskAnnotator(
    color=custom_palette,
    opacity=1.0,
    color_lookup=sv.ColorLookup.CLASS
)

# ==========================================
# 3. Background Threads (Vision + Odometry)
# ==========================================
latest_frame = None
latest_detections = None
lock = threading.Lock()
running = True

def ai_worker():
    global latest_frame, latest_detections, running
    while running:
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()

        if frame_to_process is not None:
            results = local_model.predict(
                source=frame_to_process, imgsz=640, conf=0.6, verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            with lock:
                latest_detections = detections
        else:
            time.sleep(0.01)

def serial_odometry_worker(arduino):
    global running, robot_x, robot_y, robot_theta, x_path, y_path
    global first_reading, prev_angle_l, prev_angle_r
    
    while running:
        if arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("E,"):
                    parts = line.split(',')
                    if len(parts) == 3:
                        curr_angle_l = float(parts[1])
                        curr_angle_r = float(parts[2])
                        
                        if first_reading:
                            prev_angle_l = curr_angle_l
                            prev_angle_r = curr_angle_r
                            first_reading = False
                            continue
                            
                        # Calculate delta angles
                        delta_angle_l = curr_angle_l - prev_angle_l
                        delta_angle_r = curr_angle_r - prev_angle_r
                        prev_angle_l = curr_angle_l
                        prev_angle_r = curr_angle_r
                        
                        # Odometry Equations
                        circumference = 2 * math.pi * WHEEL_RADIUS_CM
                        dl = circumference * (delta_angle_l / 360.0)
                        dr = circumference * (delta_angle_r / 360.0)
                        
                        d = (dr + dl) / 2.0
                        delta_theta = (dr - dl) / TRACK_WIDTH_CM
                        
                        robot_theta += delta_theta
                        robot_x += d * math.cos(robot_theta)
                        robot_y += d * math.sin(robot_theta)
                        
                        with lock:
                            x_path.append(robot_x)
                            y_path.append(robot_y)
            except Exception:
                pass
        else:
            time.sleep(0.01)

# ==========================================
# 4. Main Hardware & RL Loop
# ==========================================
def run_robot():
    global latest_frame, latest_detections, running

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_model = DQN().to(device)
    dqn_model.load_state_dict(torch.load(BRAIN_PATH, map_location=device))
    dqn_model.eval()

    try:
        arduino = serial.Serial('COM12', 115200, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"Could not open Serial Port: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    arduino.reset_input_buffer()
    
    # Wait for initial sync
    synced = False
    sync_start = time.time()
    while time.time() - sync_start < 10.0:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("E,"):
                synced = True
                break

    if not synced:
        print("ERROR: No sync line from OpenCR.")
        running = False
        return

    # Start Background Threads
    threading.Thread(target=ai_worker, daemon=True).start()
    threading.Thread(target=serial_odometry_worker, args=(arduino,), daemon=True).start()

    frame_count = 0
    forward_speed = 15  
    turn_diff = 5     

    prev_time = time.time()
    last_serial_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with lock:
                latest_frame = frame
                detections = latest_detections

            display_frame = np.zeros_like(frame)
            display_frame[:] = (35, 35, 35)

            if detections is not None and len(detections) > 0 and detections.mask is not None:
                display_frame = mask_annotator.annotate(scene=display_frame, detections=detections)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            if curr_time - last_serial_time >= 0.1:
                last_serial_time = curr_time

                state = preprocess_frame(display_frame).to(device)
                with torch.no_grad():
                    action = dqn_model(state).argmax().item()

                action_text = "LEFT" if action == 2 else "RIGHT" if action == 0 else "STRAIGHT"
                color = (0, 255, 0) if action == 1 else (0, 165, 255)
                frame_count += 1

                if action == 2:  # LEFT TURN
                    left_speed = -(forward_speed - turn_diff)
                    right_speed = -(forward_speed + turn_diff)
                elif action == 0:  # RIGHT TURN
                    left_speed = -(forward_speed + turn_diff)
                    right_speed = -(forward_speed - turn_diff)
                elif action == 1:  # STRAIGHT
                    left_speed = -forward_speed
                    right_speed = -forward_speed

                command = f"{int(left_speed)},{int(right_speed)}\n"
                
                # Removed arduino.reset_input_buffer() to protect telemetry data
                arduino.reset_output_buffer()
                arduino.write(command.encode('utf-8'))
                arduino.flush()

                cv2.putText(display_frame, f"AI: {action_text} | FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            small_display = cv2.resize(display_frame, (640, 640))
            cv2.imshow("Robot Vision & AI", small_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nForce quit detected!")
    finally:
        running = False
        if 'arduino' in locals() and arduino.is_open:
            try:
                arduino.reset_output_buffer()
                arduino.write("0,0\n".encode('utf-8'))
                arduino.flush()
                time.sleep(0.05)
            except Exception:
                pass
            arduino.close()
            
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        
        # --- PLOT ODOMETRY MAP ---
        print("\nGenerating Odometry Map...")
        plt.figure(figsize=(8, 8))
        with lock:
            plt.plot(x_path, y_path, marker='o', linestyle='-', color='b', markersize=2, label='AI Path')
            plt.plot(x_path[0], y_path[0], marker='s', color='g', markersize=8, label='Start')
            plt.plot(x_path[-1], y_path[-1], marker='X', color='r', markersize=8, label='End')
            
        plt.title('Autonomous RL Robot Odometry')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.grid(True)
        plt.axis('equal') 
        plt.legend()
        plt.show()

if __name__ == "__main__":
    run_robot()
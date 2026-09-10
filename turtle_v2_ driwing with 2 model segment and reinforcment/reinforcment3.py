import cv2
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import serial
import os
from ultralytics import YOLO
import supervision as sv

# ==========================================
# 0. Paths (robust to working directory)
# ==========================================
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
print(f"Loading local AI model from: {MODEL_PATH}...")

if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: best.pt not found at {MODEL_PATH}")
    exit()

try:
    local_model = YOLO(MODEL_PATH)
    print("Local Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

print(f"Model class mapping: {local_model.names}")

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
# 3. Background AI Worker Thread
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
                source=frame_to_process,
                imgsz=640,
                conf=0.6,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            with lock:
                latest_detections = detections
        else:
            time.sleep(0.01)

ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

# ==========================================
# 4. Main Hardware & RL Loop
# ==========================================
def run_robot():
    global latest_frame, latest_detections, running

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running PyTorch DQN on: {device}")

    dqn_model = DQN().to(device)
    if not os.path.isfile(BRAIN_PATH):
        print(f"Error: 'robot_brain.pth' not found at {BRAIN_PATH}.")
        running = False
        return

    dqn_model.load_state_dict(torch.load(BRAIN_PATH, map_location=device))
    dqn_model.eval()
    print("Successfully loaded PyTorch AI weights!")

    try:
        arduino = serial.Serial('COM12', 115200, timeout=1)
        time.sleep(2)
        print("Connected to OpenCR on COM12!")
    except Exception as e:
        print(f"Could not open Serial Port: {e}")
        running = False
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        running = False
        return

    print("Checking connection with OpenCR...")
    arduino.reset_input_buffer()

    sync_timeout_s = 10.0
    sync_start = time.time()
    synced = False

    while time.time() - sync_start < sync_timeout_s:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("E,"):
                print("Synced! OpenCR is ready in Velocity Mode.")
                synced = True
                break

    if not synced:
        print("ERROR: Did not receive sync line 'E,...' from OpenCR.")
        arduino.close()
        cap.release()
        running = False
        return

    print("\n--- SYSTEM READY. Press 'q' or Ctrl+C to quit. ---")
    frame_count = 0

    # VELOCITY SETTINGS (RPM)
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

            # Update speeds 10 times a second (0.1s) for fast response
            if curr_time - last_serial_time >= 0.1:
                last_serial_time = curr_time

                state = preprocess_frame(display_frame).to(device)
                with torch.no_grad():
                    action = dqn_model(state).argmax().item()

                action_text = "LEFT" if action == 2 else "RIGHT" if action == 0 else "STRAIGHT"
                color = (0, 255, 0) if action == 1 else (0, 165, 255)
                frame_count += 1

                # Calculate Speed directly (no longer adding to previous angles)
                # Maintaining negative signs from previous logic to ensure it moves the same direction
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
                
                # PURGE QUEUES: Dump old data instantly before writing
                arduino.reset_output_buffer()
                arduino.reset_input_buffer()
                
                # FORCE SEND: Send the command and flush to OS
                arduino.write(command.encode('utf-8'))
                arduino.flush()

                print(f"Frame {frame_count} - Action: {action_text} | Speed: {command.strip()}")

                cv2.putText(display_frame, f"AI: {action_text} | FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            small_display = cv2.resize(display_frame, (640, 640))
            cv2.imshow("Robot Vision & AI (320x320 view)", small_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nForce quit detected (Ctrl+C)!")

    finally:
        print("\nInitiating emergency shutdown...")
        running = False

        if 'arduino' in locals() and arduino.is_open:
            try:
                # Stop Motors (0 Velocity)
                arduino.reset_output_buffer()
                stop_cmd = "0,0\n"
                arduino.write(stop_cmd.encode('utf-8'))
                arduino.flush()
                time.sleep(0.05)
            except Exception:
                pass
            arduino.close()
            print("Motors disconnected and stopped.")

        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("Camera released.")

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("UI closed. Shutdown complete.")
        os._exit(0)

if __name__ == "__main__":
    run_robot()
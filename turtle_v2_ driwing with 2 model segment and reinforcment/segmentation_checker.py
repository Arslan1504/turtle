"""
Real-time segmentation on video capture using a YOLO segmentation model (best.pt).

- Model file: best.pt (must be in the SAME folder as this script)
- Inference resolution: 640x640 (matches training size)
- Display resolution: 320x320 (resized after prediction for a smaller window)

Requirements:
    pip install ultralytics opencv-python

Usage:
    python segment_video.py            # uses webcam (source=0)
    python segment_video.py my_video.mp4   # uses a video file instead
"""

import os
import sys
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_aug.pt")   # best.pt must sit next to this script

INFER_SIZE = 640     # model was trained at 640, so predict at 640
DISPLAY_SIZE = 320    # resize the output before showing it

CONF_THRES = 0.25     # confidence threshold, tweak as needed


def main():
    # ---- Load model -------------------------------------------------------
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Could not find model at: {MODEL_PATH}")
        print("Make sure best.pt is in the same folder as this script.")
        sys.exit(1)

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ---- Open video source --------------------------------------------
    # default: webcam (0). If a path/filename is passed as an argument, use that instead.
    source = sys.argv[1] if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        sys.exit(1)

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / failed to read frame.")
            break

        # ---- Predict segmentation at 640x640 --------------------------
        results = model.predict(
            source=frame,
            imgsz=INFER_SIZE,
            conf=CONF_THRES,
            verbose=False,
        )

        # results[0].plot() draws masks/boxes/labels on the frame (at inference size)
        annotated = results[0].plot()

        # ---- Resize down to 320x320 for display ------------------------
        display_frame = cv2.resize(annotated, (DISPLAY_SIZE, DISPLAY_SIZE))

        cv2.imshow("Segmentation (320x320)", display_frame)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
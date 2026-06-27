"""Diagnose webcam - find working camera index, backend, and format."""
import cv2
import numpy as np
import time

backends = [
    ("Default", cv2.CAP_ANY),
    ("DirectShow", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
]

fourcc_options = [
    ("Auto", None),
    ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
    ("YUY2", cv2.VideoWriter_fourcc(*"YUY2")),
]

print("=" * 60)
print("  WEBCAM DIAGNOSTIC")
print("=" * 60)

for cam_idx in range(3):
    for backend_name, backend in backends:
        for fourcc_name, fourcc in fourcc_options:
            try:
                cap = cv2.VideoCapture(cam_idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                if fourcc is not None:
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Read and discard warmup frames
                for _ in range(30):
                    cap.read()
                    time.sleep(0.02)

                ret, frame = cap.read()
                if ret and frame is not None:
                    mean_px = np.mean(frame)
                    is_black = mean_px < 10
                    is_dark = mean_px < 50
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

                    status = "BLACK" if is_black else ("DARK" if is_dark else "OK")
                    marker = "  <<<< BEST" if not is_dark else ""
                    print(f"Camera {cam_idx} | {backend_name:11s} | {fourcc_name:4s} | "
                          f"{w}x{h} | format={fourcc_str} | mean={mean_px:.0f} | "
                          f"{status}{marker}")
                else:
                    print(f"Camera {cam_idx} | {backend_name:11s} | {fourcc_name:4s} | "
                          f"FAILED to read frame")

                cap.release()
            except Exception as e:
                print(f"Camera {cam_idx} | {backend_name:11s} | {fourcc_name:4s} | ERROR: {e}")
                try:
                    cap.release()
                except:
                    pass

print("\n" + "=" * 60)
print("  Look for lines marked '<<<< BEST' above.")
print("  Those camera/backend/format combos produce visible images.")
print("=" * 60)

#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from datetime import datetime

print("Lepton 3.0 - Ubuntu 160x120 Y16 Mode (True 14-bit visualization + AVI Recording)")

# -----------------------------
# Key Controls Guide
# -----------------------------
print("\n📖 Key Controls:")
print("  v - Start / Stop AVI recording (silent)")
print("  s - Take screenshot (PNG)")
print("  m - Change colormap")
print("  h - Toggle HUD display")
print("  r - Toggle raw/grayscale view")
print("  q - Quit\n")

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=4)
args = parser.parse_args()
device = f"/dev/video{args.device}"

# -----------------------------
# Open Camera
# -----------------------------
cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1','6',' '))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

# -----------------------------
# Parameters
# -----------------------------
width, height = 160, 120
scale = 4
hud = True
raw_mode = False

colormap = 3
maps = [
    cv2.COLORMAP_JET,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_VIRIDIS
]

# -----------------------------
# Recording Variables
# -----------------------------
recording = False
video_writer = None
fps = 9
frame_counter = 0
current_filename = ""

cv2.namedWindow("Thermal", cv2.WINDOW_NORMAL)

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    if len(frame.shape) == 3:
        frame = frame[:, :, 0]

    frame = frame.astype(np.uint16)

    # -----------------------------
    # Temperature Conversion
    # -----------------------------
    CAL_GAIN = 0.033345
    CAL_OFFSET = -241.95

    temp_c = frame.astype(np.float32) * CAL_GAIN + CAL_OFFSET

    center_temp = temp_c[height//2, width//2]
    min_temp = float(np.min(temp_c))
    max_temp = float(np.max(temp_c))

    # -----------------------------
    # Visualization
    # -----------------------------
    if raw_mode:
        disp = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        disp = cv2.convertScaleAbs(disp)
        disp = cv2.resize(disp, (width*scale, height*scale),
                          interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    else:
        norm = (frame.astype(np.float32) - frame.min()) / (frame.max() - frame.min() + 1)
        norm8 = np.uint8(norm * 255)
        disp = cv2.resize(norm8, (width*scale, height*scale),
                          interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(disp, maps[colormap])

    # -----------------------------
    # Crosshair
    # -----------------------------
    cx, cy = width*scale//2, height*scale//2
    cv2.line(heatmap, (cx-20, cy), (cx+20, cy), (255,255,255), 2)
    cv2.line(heatmap, (cx, cy-20), (cx, cy+20), (255,255,255), 2)
    cv2.putText(heatmap, f"{center_temp:.2f} C", (cx+10, cy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # -----------------------------
    # HUD
    # -----------------------------
    if hud:
        cv2.rectangle(heatmap, (0,0), (220,80), (0,0,0), -1)
        cv2.putText(heatmap, f"Min: {min_temp:.2f} C", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(heatmap, f"Max: {max_temp:.2f} C", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(heatmap, f"Center: {center_temp:.2f} C", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # -----------------------------
    # Show window
    # -----------------------------
    cv2.imshow("Thermal", heatmap)

    # -----------------------------
    # Write Recording
    # -----------------------------
    if recording and video_writer is not None:
        video_writer.write(heatmap)
        frame_counter += 1

    # -----------------------------
    # Key Controls
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('m'):
        colormap = (colormap + 1) % len(maps)

    if key == ord('h'):
        hud = not hud

    if key == ord('r'):
        raw_mode = not raw_mode

    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, heatmap)
        print(f"✅ Screenshot saved: {filename}")

    if key == ord('v'):
        recording = not recording

        if recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_filename = f"thermal_{timestamp}.avi"

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                current_filename,
                fourcc,
                fps,
                (width*scale, height*scale)
            )

            frame_counter = 0

        else:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

            print(f"✅ Video saved: {current_filename} ({frame_counter} frames)")

# -----------------------------
# Cleanup
# -----------------------------
cap.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()


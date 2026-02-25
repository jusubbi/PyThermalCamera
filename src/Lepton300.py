#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from datetime import datetime

print("Lepton 3.0 - FLIR E-Series UI Edition")

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

width, height = 160, 120
scale = 4

# -----------------------------
# Auto Scale
# -----------------------------
auto_min = None
auto_max = None
ALPHA = 0.03
MARGIN = 1.0

# Colormap options
colormaps = [
    cv2.COLORMAP_JET,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_VIRIDIS
]
colormap_index = 3  # start with INFERNO

# -----------------------------
# Recording / HUD / Raw
# -----------------------------
recording = False
video_writer = None
fps = 9
frame_counter = 0
current_filename = ""
hud = True
raw_mode = False

cv2.namedWindow("Thermal", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if len(frame.shape) == 3:
        frame = frame[:, :, 0]

    frame = frame.astype(np.uint16)

    # Temperature conversion
    CAL_GAIN = 0.033345
    CAL_OFFSET = -241.95
    temp_c = frame.astype(np.float32) * CAL_GAIN + CAL_OFFSET

    # Auto scale
    frame_min = float(np.min(temp_c)) - MARGIN
    frame_max = float(np.max(temp_c)) + MARGIN

    if auto_min is None:
        auto_min = frame_min
        auto_max = frame_max
    else:
        auto_min = (1 - ALPHA) * auto_min + ALPHA * frame_min
        auto_max = (1 - ALPHA) * auto_max + ALPHA * frame_max

    if auto_max - auto_min < 0.1:
        auto_max = auto_min + 0.1

    # -----------------------------
    # Normalize for display
    # -----------------------------
    if raw_mode:
        # Grayscale display
        norm = (frame.astype(np.float32) - frame.min()) / (frame.max() - frame.min() + 1)
        norm8 = np.uint8(norm * 255)
        disp = cv2.resize(norm8, (width*scale, height*scale), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    else:
        # Color map display
        norm = np.clip((temp_c - auto_min) / (auto_max - auto_min), 0, 1)
        norm8 = np.uint8(norm * 255)
        disp = cv2.resize(norm8, (width*scale, height*scale), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(disp, colormaps[colormap_index])

    h, w = heatmap.shape[:2]

    # -----------------------------
    # Crosshair
    # -----------------------------
    cx, cy = w//2, h//2
    cv2.line(heatmap, (cx-15, cy), (cx+15, cy), (255,255,255), 1)
    cv2.line(heatmap, (cx, cy-15), (cx, cy+15), (255,255,255), 1)
    cv2.circle(heatmap, (cx, cy), 4, (255,255,255), 1)

    center_temp = temp_c[height//2, width//2]

    if hud:
        # Center temp text
        txt = f"{center_temp:.1f} C"
        cv2.putText(heatmap, txt, (cx+20, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(heatmap, txt, (cx+20, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # -----------------------------
    # Color bar (with temps)
    # -----------------------------
    bar_width = 30
    bar_height = h - 120
    bar_x = w - 60
    bar_y = 60

    # Gradient
    for i in range(bar_height):
        ratio = 1 - (i / bar_height)
        color_val = int(ratio * 255)
        if raw_mode:
            color = [color_val]*3
        else:
            color = cv2.applyColorMap(np.uint8([[color_val]]), colormaps[colormap_index])[0][0].tolist()
        cv2.line(heatmap, (bar_x, bar_y+i), (bar_x+bar_width, bar_y+i), color, 1)

    # Border
    cv2.rectangle(heatmap, (bar_x-1, bar_y-1), (bar_x+bar_width+1, bar_y+bar_height+1), (255,255,255), 1)
    cv2.putText(heatmap, f"{auto_max:.0f}", (bar_x-40, bar_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(heatmap, f"{auto_min:.0f}", (bar_x-40, bar_y+bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Thermal", heatmap)

    # -----------------------------
    # Recording
    # -----------------------------
    if recording and video_writer is not None:
        video_writer.write(heatmap)
        frame_counter += 1

    # -----------------------------
    # Key controls
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        colormap_index = (colormap_index + 1) % len(colormaps)
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
            video_writer = cv2.VideoWriter(current_filename, fourcc, fps, (width*scale, height*scale))
            frame_counter = 0
            print("🎬 Recording started...")
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

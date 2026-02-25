#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from datetime import datetime

print("Lepton 3.0 - FLIR E-Series UI Edition")

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
# Stable Auto Scale
# -----------------------------
auto_min = None
auto_max = None
ALPHA = 0.03
MARGIN = 1.0

colormap = cv2.COLORMAP_INFERNO

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

    # Normalize using stable scale
    norm = np.clip((temp_c - auto_min) / (auto_max - auto_min), 0, 1)
    norm8 = np.uint8(norm * 255)

    disp = cv2.resize(norm8, (width*scale, height*scale),
                      interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(disp, colormap)

    h, w = heatmap.shape[:2]

    # -----------------------------
    # Crosshair (E-series style)
    # -----------------------------
    cx, cy = w//2, h//2
    cv2.line(heatmap, (cx-15, cy), (cx+15, cy), (255,255,255), 1)
    cv2.line(heatmap, (cx, cy-15), (cx, cy+15), (255,255,255), 1)
    cv2.circle(heatmap, (cx, cy), 4, (255,255,255), 1)

    center_temp = temp_c[height//2, width//2]

    # Center temp text (with shadow)
    txt = f"{center_temp:.1f}°C"
    cv2.putText(heatmap, txt, (cx+20, cy+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(heatmap, txt, (cx+20, cy+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # -----------------------------
    # Min/Max Labels (black boxes)
    # -----------------------------
    max_temp = auto_max
    min_temp = auto_min

    def draw_label(img, text, x, y):
        (tw, th), _ = cv2.getTextSize(text,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7, 2)
        cv2.rectangle(img, (x, y-th-10),
                      (x+tw+10, y+5),
                      (0,0,0), -1)
        cv2.putText(img, text, (x+5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,255), 2)

    draw_label(heatmap, f"{max_temp:.1f}°C", 10, 30)
    draw_label(heatmap, f"{min_temp:.1f}°C", w-150, 30)

    # -----------------------------
    # Single Vertical Color Bar
    # -----------------------------
    bar_width = 30
    bar_height = h - 120
    bar_x = w - 60
    bar_y = 60

    # Black frame
    cv2.rectangle(heatmap,
                  (bar_x-3, bar_y-3),
                  (bar_x+bar_width+3, bar_y+bar_height+3),
                  (0,0,0), -1)

    # Gradient (TOP = HOT)
    for i in range(bar_height):
        ratio = 1 - (i / bar_height)
        color_val = int(ratio * 255)
        color = cv2.applyColorMap(
            np.uint8([[color_val]]),
            colormap)[0][0].tolist()

        cv2.line(heatmap,
                 (bar_x, bar_y+i),
                 (bar_x+bar_width, bar_y+i),
                 color, 1)

    # Border
    cv2.rectangle(heatmap,
                  (bar_x-3, bar_y-3),
                  (bar_x+bar_width+3, bar_y+bar_height+3),
                  (255,255,255), 1)

    # Scale numbers
    cv2.putText(heatmap, f"{max_temp:.0f}",
                (bar_x-45, bar_y+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1)

    cv2.putText(heatmap, f"{min_temp:.0f}",
                (bar_x-45, bar_y+bar_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1)

    cv2.imshow("Thermal", heatmap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

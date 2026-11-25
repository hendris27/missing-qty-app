
import ctypes
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

SAVE_DIR = Path("dataset/images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera tidak ditemukan. Coba CAMERA_ID = 1.")
    raise SystemExit(1)

print("Kamera aktif.")
print("Tekan S untuk menyimpan foto.")
print("Tekan Q untuk keluar.")

window_name = "Capture Tray PCB"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)
screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca kamera.")
        break

    display = np.full(
        (screen_height, screen_width, 3),
        (195, 114, 70),
        dtype=np.uint8
    )
    outer_margin = max(8, round(screen_height * 0.02))
    border_color = (35, 48, 70)
    one_centimeter = round(ctypes.windll.user32.GetDpiForSystem() / 2.54)
    camera_right = round(screen_width * 0.56) - 4 * one_centimeter

    cv2.rectangle(
        display,
        (outer_margin, outer_margin),
        (screen_width - outer_margin, screen_height - outer_margin),
        border_color,
        1
    )
    cv2.rectangle(
        display,
        (outer_margin + 1, outer_margin + 1),
        (camera_right, screen_height - outer_margin - 1),
        (255, 255, 255),
        cv2.FILLED
    )

    camera_padding = max(8, round(screen_height * 0.01))
    camera_x = outer_margin + camera_padding
    camera_y = outer_margin + camera_padding
    camera_width = camera_right - camera_x - camera_padding
    camera_height = screen_height - outer_margin - camera_y - camera_padding
    frame_height, frame_width = frame.shape[:2]
    scale = max(camera_width / frame_width, camera_height / frame_height)
    resized_width = max(camera_width, round(frame_width * scale))
    resized_height = max(camera_height, round(frame_height * scale))
    resized_frame = cv2.resize(frame, (resized_width, resized_height))
    crop_x = (resized_width - camera_width) // 2
    crop_y = (resized_height - camera_height) // 2
    camera_view = resized_frame[
        crop_y:crop_y + camera_height,
        crop_x:crop_x + camera_width
    ]
    display[
        camera_y:camera_y + camera_height,
        camera_x:camera_x + camera_width
    ] = camera_view

    qty_left = round(screen_width * 0.69)
    qty_top = round(screen_height * 0.25)
    qty_right = round(screen_width * 0.89)
    qty_bottom = round(screen_height * 0.51)
    cv2.rectangle(
        display,
        (qty_left, qty_top),
        (qty_right, qty_bottom),
        (105, 171, 70),
        cv2.FILLED
    )

    qty_text = f"QTY : {count}"
    qty_scale = screen_height / 558 * 0.7
    qty_thickness = max(1, round(screen_height / 558 * 2))
    (qty_text_width, qty_text_height), _ = cv2.getTextSize(
        qty_text, cv2.FONT_HERSHEY_SIMPLEX, qty_scale, qty_thickness
    )
    cv2.putText(
        display,
        qty_text,
        (qty_left + (qty_right - qty_left - qty_text_width) // 2,
         qty_top + (qty_bottom - qty_top + qty_text_height) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        qty_scale,
        (255, 255, 255),
        qty_thickness,
        cv2.LINE_AA
    )

    frame_text = ""
    frame_scale = screen_height / 558 * 0.7
    frame_thickness = max(1, round(screen_height / 558 * 2))
    (frame_text_width, frame_text_height), _ = cv2.getTextSize(
        frame_text, cv2.FONT_HERSHEY_SIMPLEX, frame_scale, frame_thickness
    )
    frame_center_x = round(screen_width * 0.8)
    frame_center_y = round(screen_height * 0.8)
    cv2.putText(
        display,
        frame_text,
        (frame_center_x - frame_text_width // 2,
         frame_center_y + frame_text_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        frame_scale,
        (0, 0, 0),
        frame_thickness,
        cv2.LINE_AA
    )

    controls_text = "S = SAVE  |  Q = QUIT"
    controls_scale = screen_height / 558 * 0.35
    controls_thickness = max(1, round(screen_height / 558))
    (controls_width, controls_height), _ = cv2.getTextSize(
        controls_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        controls_scale,
        controls_thickness
    )
    controls_center_x = round(screen_width * 0.8)
    controls_y = round(screen_height * 0.94)
    cv2.putText(
        display,
        controls_text,
        (controls_center_x - controls_width // 2, controls_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        controls_scale,
        (255, 255, 255),
        controls_thickness,
        cv2.LINE_AA
    )

    cv2.imshow(window_name, display)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("s"), ord("S")):
        filename = datetime.now().strftime(
            "tray_%Y%m%d_%H%M%S_%f.jpg"
        )
        path = SAVE_DIR / filename
        camera_aspect = camera_width / camera_height
        if frame_width / frame_height > camera_aspect:
            saved_width = round(frame_height * camera_aspect)
            saved_left = (frame_width - saved_width) // 2
            saved_frame = frame[:, saved_left:saved_left + saved_width]
        else:
            saved_height = round(frame_width / camera_aspect)
            saved_top = (frame_height - saved_height) // 2
            saved_frame = frame[saved_top:saved_top + saved_height, :]

        if cv2.imwrite(str(path), saved_frame):
            count += 1
            print(f"Foto tersimpan: {path}")
        else:
            print("Gagal menyimpan foto.")

    elif key in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Pengambilan selesai. Foto baru disimpan: {count}")

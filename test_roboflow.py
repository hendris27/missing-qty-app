import threading
import ctypes
import os
import queue
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_DIRECTORY = Path(__file__).resolve().parent
# Daftar model: (nama di layar, nama file .pt, target QTY).
# Ubah nama tampilan atau file di baris yang sesuai.
# Ubah target QTY pada baris model yang sesuai.
# Untuk menambah model, tambahkan baris baru di akhir daftar ini.
MODEL_CONFIG = [
    ("Model 1", "best.pt", 40),
    ("A8560/6732", "A85606732.pt", 36),
    ("Model 3", "best2.pt", 40),
    ("Model 4", "best3.pt", 40),
    ("Model 5", "best4.pt", 40),
    ("Model 6", "best5.pt", 40),
    ("Model 7", "best6.pt", 40),
    ("Model 8", "best7.pt", 40),
    ("Model 9", "best8.pt", 40),
    ("Model 10", "best9.pt", 40),
]
MODEL_OPTIONS = tuple(range(1, len(MODEL_CONFIG) + 1))
# Jika YOLO_MODEL_PATH diatur, path tersebut menggantikan file Model 1.
MODEL_1_PATH_OVERRIDE = os.environ.get("YOLO_MODEL_PATH")
MODEL_PATH = (
    Path(MODEL_1_PATH_OVERRIDE)
    if MODEL_1_PATH_OVERRIDE
    else MODEL_DIRECTORY / MODEL_CONFIG[0][1]
)
INFERENCE_INTERVAL_SECONDS = 0.25
TRAY_REMOVAL_CONFIRM_SECONDS = 1.0
COUNT_RESUME_STABLE_SECONDS = 1.0
MOTION_PIXEL_DIFF_THRESHOLD = 20
MOTION_RATIO_THRESHOLD = 0.02
current_qty = 0
current_target_qty = 0
current_detections = []
tray_removal_started_at = None
tray_removal_alerted = False
counting_enabled = True
counting_generation = 0
state_lock = threading.Lock()
model_lock = threading.Lock()
stop_event = threading.Event()
tray_removal_event = threading.Event()
inference_queue = queue.Queue(maxsize=1)

model = None
selected_model_number = None
model_selection_locked = False
model_dropdown_open = False
model_status_message = ""
model_status_until = 0.0


winmm = ctypes.windll.winmm
winmm.mciSendStringW.argtypes = [
    ctypes.c_wchar_p,
    ctypes.c_wchar_p,
    ctypes.c_uint,
    ctypes.c_void_p,
]
winmm.mciSendStringW.restype = ctypes.c_uint
winmm.mciGetErrorStringW.argtypes = [
    ctypes.c_uint,
    ctypes.c_wchar_p,
    ctypes.c_uint,
]
winmm.mciGetErrorStringW.restype = ctypes.c_int


def send_mci_command(command):
    error_code = winmm.mciSendStringW(command, None, 0, None)
    if error_code:
        error_message = ctypes.create_unicode_buffer(256)
        winmm.mciGetErrorStringW(error_code, error_message, len(error_message))
        print(f"Audio alert gagal ({error_code}): {error_message.value}")
        return False
    return True


def inference_worker():
    global current_qty, current_detections
    global tray_removal_started_at, tray_removal_alerted
    while not stop_event.is_set():
        try:
            frame, generation = inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            with model_lock:
                active_model = model
                if active_model is None:
                    continue
                result = active_model(frame, verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    center_x, center_y, box_width, box_height = (
                        box.xywh[0].tolist()
                    )
                    detections.append(
                        {
                            "x": center_x,
                            "y": center_y,
                            "width": box_width,
                            "height": box_height,
                        }
                    )
            with state_lock:
                if not counting_enabled or generation != counting_generation:
                    continue
                if detections:
                    tray_removal_started_at = None
                    tray_removal_alerted = False
                elif current_qty >= current_target_qty:
                    tray_removal_started_at = None
                    tray_removal_alerted = False
                elif 0 < current_qty < current_target_qty:
                    tray_removal_started_at = time.monotonic()
                elif (
                    tray_removal_started_at is not None
                    and not tray_removal_alerted
                    and time.monotonic() - tray_removal_started_at
                    >= TRAY_REMOVAL_CONFIRM_SECONDS
                ):
                    tray_removal_event.set()
                    tray_removal_alerted = True
                current_qty = len(detections)
                current_detections = detections
        except Exception as error:
            print(f"YOLO inference gagal: {error}")


def require_tray_removal_password():
    dialog = tk.Tk()
    dialog.title("Missing qty detected")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)

    tk.Label(
        dialog,
        text="Missing qty detected\nInput password to open:",
        padx=20,
        pady=10,
    ).pack()
    password = tk.StringVar()
    password_entry = tk.Entry(dialog, textvariable=password, show="*", width=12)
    password_entry.pack(padx=20, pady=(0, 8))
    status = tk.Label(dialog, text="", fg="red")
    status.pack(pady=(0, 10))

    def check_password(_event=None):
        value = password.get()
        if len(value) == 2:
            if value == "00":
                dialog.destroy()
            else:
                password.set("")
                status.config(text="Password salah")

    password_entry.bind("<KeyRelease>", check_password)
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    dialog.update_idletasks()
    window_width = dialog.winfo_width()
    window_height = dialog.winfo_height()
    position_x = (dialog.winfo_screenwidth() - window_width) // 2
    position_y = (dialog.winfo_screenheight() - window_height) // 2
    dialog.geometry(f"+{position_x}+{position_y}")
    dialog.deiconify()
    dialog.update()
    dialog.lift()
    dialog.focus_force()
    password_entry.focus_force()
    dialog.grab_set()
    dialog.wait_window()


def require_model_change_password():
    dialog = tk.Tk()
    dialog.withdraw()
    dialog.title("Ganti model")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)

    tk.Label(
        dialog,
        text="Masukkan password untuk mengganti model:",
        padx=20,
        pady=10,
    ).pack()
    password = tk.StringVar()
    password_entry = tk.Entry(dialog, textvariable=password, show="*", width=12)
    password_entry.pack(padx=20, pady=(0, 8))
    status = tk.Label(dialog, text="", fg="red")
    status.pack(pady=(0, 10))
    authorized = {"value": False}

    def check_password(_event=None):
        value = password.get()
        if len(value) >= 4:
            if value == "0000":
                authorized["value"] = True
                dialog.destroy()
            else:
                password.set("")
                status.config(text="Password salah")

    password_entry.bind("<KeyRelease>", check_password)
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.update_idletasks()
    window_width = dialog.winfo_width()
    window_height = dialog.winfo_height()
    position_x = (dialog.winfo_screenwidth() - window_width) // 2
    position_y = (dialog.winfo_screenheight() - window_height) // 2
    dialog.geometry(f"+{position_x}+{position_y}")
    dialog.deiconify()
    dialog.update()
    dialog.lift()
    dialog.focus_force()
    password_entry.focus_force()
    dialog.grab_set()
    dialog.wait_window()
    return authorized["value"]


def model_label_for_number(model_number):
    if model_number is None:
        return "Belum ada model"
    if 1 <= model_number <= len(MODEL_CONFIG):
        return MODEL_CONFIG[model_number - 1][0]
    return f"Model {model_number}"


def model_path_for_number(model_number):
    if not 1 <= model_number <= len(MODEL_CONFIG):
        return None
    if model_number == 1 and MODEL_1_PATH_OVERRIDE:
        return MODEL_PATH
    return MODEL_DIRECTORY / MODEL_CONFIG[model_number - 1][1]


def set_model_status(message, duration=4.0):
    global model_status_message, model_status_until
    model_status_message = message
    model_status_until = time.monotonic() + duration


def select_model(model_number):
    global model, selected_model_number, model_selection_locked
    global model_dropdown_open, current_qty, current_detections
    global counting_generation, tray_removal_started_at, tray_removal_alerted
    global current_target_qty

    model_path = model_path_for_number(model_number)
    model_label = model_label_for_number(model_number)
    if model_path is None:
        set_model_status(f"{model_label} belum terdaftar")
        model_dropdown_open = True
        return False
    if not model_path.is_file():
        set_model_status(f"File model tidak ditemukan: {model_path.name}")
        model_dropdown_open = True
        return False

    with model_lock:
        active_model = model
    if model_number == selected_model_number and active_model is not None:
        with state_lock:
            current_target_qty = MODEL_CONFIG[model_number - 1][2]
        model_selection_locked = True
        model_dropdown_open = False
        set_model_status(f"{model_label} dipilih")
        return True

    try:
        loaded_model = YOLO(str(model_path))
        with model_lock:
            with state_lock:
                model = loaded_model
                counting_generation += 1
                current_target_qty = MODEL_CONFIG[model_number - 1][2]
                current_qty = 0
                current_detections = []
                tray_removal_started_at = None
                tray_removal_alerted = False
        tray_removal_event.clear()
        selected_model_number = model_number
        model_selection_locked = True
        model_dropdown_open = False
        set_model_status(f"{model_label} aktif")
        print(f"{model_label} dimuat: {model_path}")
        return True
    except Exception as error:
        print(f"Gagal memuat {model_label} ({model_path}): {error}")
        set_model_status(f"Gagal memuat {model_label}")
        model_dropdown_open = True
        return False


def draw_rounded_rectangle(image, box, color, radius):
    x1, y1, x2, y2 = map(int, box)
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, cv2.FILLED)
        return

    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
    for center in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(image, center, radius, color, cv2.FILLED, cv2.LINE_AA)


def fill_rounded_gradient(image, box, top_color, bottom_color, radius):
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    if width <= 0 or height <= 0:
        return

    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top = np.asarray(top_color, dtype=np.float32).reshape(1, 1, 3)
    bottom = np.asarray(bottom_color, dtype=np.float32).reshape(1, 1, 3)
    gradient_rows = (top * (1.0 - blend) + bottom * blend).astype(np.uint8)
    gradient = np.broadcast_to(gradient_rows, (height, width, 3))
    mask = np.zeros((height, width), dtype=np.uint8)
    draw_rounded_rectangle(
        mask,
        (0, 0, width - 1, height - 1),
        255,
        min(radius, width // 2, height // 2),
    )
    region = image[y1:y2 + 1, x1:x2 + 1]
    region[mask > 0] = gradient[mask > 0]


def draw_monitor_icon(image, center, size, color, thickness=2):
    center_x, center_y = center
    width = round(size * 1.25)
    height = round(size * 0.78)
    left = center_x - width // 2
    top = center_y - height // 2 - round(size * 0.08)
    right = left + width
    bottom = top + height
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)
    cv2.line(
        image,
        (center_x, bottom),
        (center_x, bottom + round(size * 0.22)),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (center_x - round(size * 0.24), bottom + round(size * 0.23)),
        (center_x + round(size * 0.24), bottom + round(size * 0.23)),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_refresh_icon(image, center, radius, color, thickness):
    center_x, center_y = center
    for arc_start, arc_end in ((195, 335), (15, 155)):
        cv2.ellipse(
            image,
            center,
            (radius, radius),
            0,
            arc_start,
            arc_end,
            color,
            thickness,
            cv2.LINE_AA,
        )

        angle = np.deg2rad(arc_end)
        radial_x = float(np.cos(angle))
        radial_y = float(np.sin(angle))
        tangent_x = -radial_y
        tangent_y = radial_x
        tip = np.array(
            [center_x + radius * radial_x, center_y + radius * radial_y],
            dtype=np.float32,
        )
        arrow_length = radius * 0.78
        arrow_half_width = radius * 0.48
        base = tip - np.array([tangent_x, tangent_y], dtype=np.float32) * arrow_length
        perpendicular = np.array([radial_x, radial_y], dtype=np.float32)
        arrow_points = np.array(
            [
                tip,
                base + perpendicular * arrow_half_width,
                base - perpendicular * arrow_half_width,
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [arrow_points], color, cv2.LINE_AA)


def draw_pcb_tray_icon(image, center, size, color, thickness=3):
    center_x, center_y = center
    half_width = round(size * 0.42)
    half_height = round(size * 0.22)
    depth = round(size * 0.28)
    top = np.array(
        [
            [center_x, center_y - half_height - depth],
            [center_x + half_width, center_y - depth],
            [center_x, center_y + half_height - depth],
            [center_x - half_width, center_y - depth],
        ],
        dtype=np.int32,
    )
    front_left = (center_x - half_width, center_y - depth)
    front_point = (center_x, center_y + half_height - depth)
    front_right = (center_x + half_width, center_y - depth)
    lower_left = (center_x - half_width, center_y + depth // 2)
    lower_point = (center_x, center_y + half_height + depth // 2)
    lower_right = (center_x + half_width, center_y + depth // 2)
    cv2.polylines(image, [top], True, color, thickness, cv2.LINE_AA)
    cv2.line(image, front_left, lower_left, color, thickness, cv2.LINE_AA)
    cv2.line(image, front_point, lower_point, color, thickness, cv2.LINE_AA)
    cv2.line(image, front_right, lower_right, color, thickness, cv2.LINE_AA)
    cv2.line(image, lower_left, lower_point, color, thickness, cv2.LINE_AA)
    cv2.line(image, lower_point, lower_right, color, thickness, cv2.LINE_AA)
    cv2.line(
        image,
        (center_x - round(size * 0.22), center_y - depth),
        (center_x + round(size * 0.22), center_y - depth),
        color,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (center_x - round(size * 0.14), center_y - depth - round(size * 0.09)),
        (center_x + round(size * 0.14), center_y - depth + round(size * 0.06)),
        color,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )


def crop_to_aspect(frame, aspect_ratio):
    frame_height, frame_width = frame.shape[:2]
    if frame_width / frame_height > aspect_ratio:
        cropped_width = max(1, round(frame_height * aspect_ratio))
        left = (frame_width - cropped_width) // 2
        return frame[:, left:left + cropped_width]

    cropped_height = max(1, round(frame_width / aspect_ratio))
    top = (frame_height - cropped_height) // 2
    return frame[top:top + cropped_height, :]


save_dir = Path("dataset/images")
save_dir.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise SystemExit("Kamera tidak ditemukan. Coba ganti camera ID.")

window_name = "Capture Tray PCB"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)
screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)
outer_margin = max(8, round(screen_height * 0.02))
camera_padding = max(8, round(screen_height * 0.01))
camera_horizontal_expand = min(
    max(1, round(screen_width * 0.01)),
    max(1, camera_padding),
)
one_centimeter = round(ctypes.windll.user32.GetDpiForSystem() / 2.54)
camera_right = (
    round(screen_width * 0.46)
    - 4 * one_centimeter
    + camera_horizontal_expand
    + 16
)
alert_path = Path(__file__).resolve().parent / "ok-alert.mp3"
alert_alias = "tray_qty_ok_alert"
alert_is_open = False
alert_is_playing = False
if alert_path.is_file():
    alert_is_open = send_mci_command(
        f'open "{alert_path}" type mpegvideo alias {alert_alias}'
    )
else:
    print(f"File audio alert tidak ditemukan: {alert_path}")
alert_can_play = alert_is_open
ng_alert_path = Path(__file__).resolve().parent / "ng-alert.mp3"
ng_alert_alias = "tray_qty_ng_alert"
ng_alert_is_open = False
ng_alert_is_playing = False
if ng_alert_path.is_file():
    ng_alert_is_open = send_mci_command(
        f'open "{ng_alert_path}" type mpegvideo alias {ng_alert_alias}'
    )
else:
    print(f"File audio alert tidak ditemukan: {ng_alert_path}")
ng_alert_can_play = ng_alert_is_open

blue_area_left = camera_right
blue_area_right = screen_width - outer_margin
blue_area_width = blue_area_right - blue_area_left
content_center_x = blue_area_left + blue_area_width // 2
qty_box_width = min(round(screen_width * 0.47), blue_area_width)
qty_center_x = content_center_x
qty_left = qty_center_x - qty_box_width // 2
qty_right = qty_left + qty_box_width
qty_top = round(screen_height * 0.20)
qty_bottom = round(screen_height * 0.80)
qty_panel_width = qty_right - qty_left
red_box_center_x = qty_left + qty_box_width // 2

header_left = qty_left
header_right = qty_right
header_top = round(screen_height * 0.025)
header_bottom = round(screen_height * 0.18)
header_gap = max(8, round(qty_box_width * 0.025))
logo_path = Path(__file__).resolve().parent / "picture" / "logo siix1.jpg"
siix_logo = cv2.imread(str(logo_path))

if siix_logo is not None:
    logo_scale = min(
        (header_bottom - header_top) / siix_logo.shape[0],
        ((header_right - header_left) * 0.42) / siix_logo.shape[1],
    )
    logo_width = max(1, round(siix_logo.shape[1] * logo_scale))
    logo_height = max(1, round(siix_logo.shape[0] * logo_scale))
    siix_logo = cv2.resize(siix_logo, (logo_width, logo_height))
    logo_y = header_top + (header_bottom - header_top - logo_height) // 2
else:
    print(f"Logo SIIX tidak ditemukan atau tidak dapat dibaca: {logo_path}")
    logo_width = 0
    logo_height = 0

header_title = "System Monitoring QTY"
header_scale = screen_height / 558 * 1.25
header_thickness = max(1, round(screen_height / 558 * 1.5))
(header_text_width, header_text_height), _ = cv2.getTextSize(
    header_title,
    cv2.FONT_HERSHEY_DUPLEX,
    header_scale,
    header_thickness,
)
title_gap = header_gap if logo_width else 0
header_text_space = qty_box_width - logo_width - title_gap
header_fit_scale = min(1.0, header_text_space / max(1, header_text_width))
header_scale *= header_fit_scale
header_thickness = max(1, round(header_thickness * header_fit_scale))
(header_text_width, header_text_height), _ = cv2.getTextSize(
    header_title,
    cv2.FONT_HERSHEY_DUPLEX,
    header_scale,
    header_thickness,
)
header_group_width = logo_width + title_gap + header_text_width
header_group_left = red_box_center_x - header_group_width // 2
logo_x = header_group_left
header_text_x = logo_x + logo_width + title_gap
header_text_y = header_top + (header_bottom - header_top + header_text_height) // 2

model_control_y = qty_bottom + round(screen_height * 0.045)
model_control_height = max(44, round(screen_height * 0.075))
model_footer_padding = max(8, round(screen_height * 0.018))
model_footer_top = model_control_y - model_footer_padding
model_footer_bottom = model_control_y + model_control_height + model_footer_padding
model_caption_area_width = round(qty_panel_width * 0.18)
control_gap = max(8, round(qty_panel_width * 0.015))
model_dropdown_width = round(qty_panel_width * 0.48)
model_change_width = round(qty_panel_width * 0.22)
model_controls_width = (
    model_caption_area_width
    + model_dropdown_width
    + model_change_width
    + 2 * control_gap
)
model_controls_left = qty_left + (qty_panel_width - model_controls_width) // 2
model_label_left = model_controls_left
model_dropdown_x = model_label_left + model_caption_area_width + control_gap
model_change_x = model_dropdown_x + model_dropdown_width + control_gap
model_option_height = max(24, round(screen_height * 0.045))
model_menu_gap = max(4, round(screen_height * 0.008))
model_menu_top = model_control_y - len(MODEL_OPTIONS) * model_option_height - model_menu_gap


def handle_model_controls(event, x, y, _flags, _parameter):
    global model_dropdown_open, model_selection_locked
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if model_dropdown_open:
        menu_bottom = model_control_y - model_menu_gap
        if (
            model_dropdown_x <= x <= model_dropdown_x + model_dropdown_width
            and model_menu_top <= y < menu_bottom
        ):
            option_index = (y - model_menu_top) // model_option_height
            if 0 <= option_index < len(MODEL_OPTIONS):
                select_model(MODEL_OPTIONS[option_index])
            return
        model_dropdown_open = False

    if (
        model_change_x <= x <= model_change_x + model_change_width
        and model_control_y <= y <= model_control_y + model_control_height
    ):
        if require_model_change_password():
            model_selection_locked = False
            model_dropdown_open = True
            set_model_status("Pilih model yang baru")
        return

    if (
        model_dropdown_x <= x <= model_dropdown_x + model_dropdown_width
        and model_control_y <= y <= model_control_y + model_control_height
    ):
        if model_selection_locked:
            set_model_status("Klik change dan masukkan password untuk membuka pilihan")
        else:
            model_dropdown_open = not model_dropdown_open


inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()
cv2.setMouseCallback(window_name, handle_model_controls)
last_inference_time = 0.0
previous_motion_frame = None
stable_since = time.monotonic()

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Gagal membaca kamera.")
            break

        display = np.full(
            (screen_height, screen_width, 3),
            (30, 18, 8),
            dtype=np.uint8,
        )
        border_color = (255, 155, 24)

        cv2.rectangle(
            display,
            (outer_margin, outer_margin),
            (screen_width - outer_margin, screen_height - outer_margin),
            border_color,
            1,
        )

        camera_x = outer_margin + camera_padding - camera_horizontal_expand
        camera_x -= 10
        camera_y = outer_margin + camera_padding - 4
        camera_width = camera_right - camera_x - camera_padding + 10
        camera_height = screen_height - outer_margin - camera_y - camera_padding
        cv2.rectangle(
            display,
            (camera_x, camera_y),
            (camera_x + camera_width - 1, camera_y + camera_height - 1),
            (255, 255, 255),
            cv2.FILLED,
        )
        camera_aspect = camera_width / camera_height
        camera_frame = crop_to_aspect(frame, camera_aspect)
        frame_height, frame_width = camera_frame.shape[:2]
        motion_frame = cv2.resize(
            cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY),
            (160, 90),
        )
        motion_detected = False
        if previous_motion_frame is not None:
            frame_difference = cv2.absdiff(
                previous_motion_frame,
                motion_frame,
            )
            changed_ratio = np.count_nonzero(
                frame_difference > MOTION_PIXEL_DIFF_THRESHOLD
            ) / frame_difference.size
            motion_detected = changed_ratio >= MOTION_RATIO_THRESHOLD
        previous_motion_frame = motion_frame

        now = time.monotonic()
        if motion_detected:
            stable_since = now
            with state_lock:
                if counting_enabled:
                    counting_enabled = False
                    counting_generation += 1
                    tray_removal_started_at = None
                    counting_paused = True
                else:
                    counting_paused = False
            if counting_paused:
                while True:
                    try:
                        inference_queue.get_nowait()
                    except queue.Empty:
                        break
        elif (
            not counting_enabled
            and now - stable_since >= COUNT_RESUME_STABLE_SECONDS
        ):
            with state_lock:
                counting_enabled = True
                counting_generation += 1

        with state_lock:
            detections_for_display = list(current_detections)
            model_selected_for_display = (
                model is not None and selected_model_number is not None
            )
            target_qty_for_display = current_target_qty if model_selected_for_display else 0
            qty_for_display = (
                min(current_qty, target_qty_for_display)
                if model_selected_for_display
                else 0
            )

        if tray_removal_event.is_set():
            tray_removal_event.clear()
            if ng_alert_can_play:
                ng_alert_is_playing = send_mci_command(
                    f"play {ng_alert_alias} repeat"
                )
                if not ng_alert_is_playing:
                    ng_alert_can_play = False
            require_tray_removal_password()
            if ng_alert_is_playing:
                send_mci_command(f"stop {ng_alert_alias}")
                ng_alert_is_playing = False

        annotated_frame = camera_frame.copy()
        for detection in detections_for_display:
            center_x = float(detection["x"])
            center_y = float(detection["y"])
            box_width = float(detection["width"])
            box_height = float(detection["height"])
            left = max(0, round(center_x - box_width / 2))
            top = max(0, round(center_y - box_height / 2))
            right = min(frame_width - 1, round(center_x + box_width / 2))
            bottom = min(frame_height - 1, round(center_y + box_height / 2))
            if right <= left or bottom <= top:
                continue

            box_color = (60, 220, 90)
            cv2.rectangle(
                annotated_frame,
                (left, top),
                (right, bottom),
                box_color,
                2,
            )

        scale = max(camera_width / frame_width, camera_height / frame_height)
        resized_width = max(camera_width, round(frame_width * scale))
        resized_height = max(camera_height, round(frame_height * scale))
        resized_frame = cv2.resize(
            annotated_frame,
            (resized_width, resized_height),
        )
        crop_x = (resized_width - camera_width) // 2
        crop_y = (resized_height - camera_height) // 2
        camera_view = resized_frame[
            crop_y:crop_y + camera_height,
            crop_x:crop_x + camera_width,
        ]
        display[
            camera_y:camera_y + camera_height,
            camera_x:camera_x + camera_width,
        ] = camera_view
        cv2.rectangle(
            display,
            (camera_x, camera_y),
            (camera_x + camera_width - 1, camera_y + camera_height - 1),
            (255, 255, 255),
            1,
        )

        header_radius = max(12, round(screen_height * 0.018))
        draw_rounded_rectangle(
            display,
            (qty_left - 5, header_top - 5, qty_right + 5, header_bottom + 5),
            (255, 155, 28),
            header_radius,
        )
        fill_rounded_gradient(
            display,
            (qty_left, header_top, qty_right, header_bottom),
            (46, 28, 12),
            (18, 14, 8),
            header_radius - 3,
        )
        cv2.line(
            display,
            (qty_left + header_radius, header_bottom - 2),
            (qty_right - header_radius, header_bottom - 2),
            (255, 190, 36),
            max(1, round(screen_height * 0.003)),
            cv2.LINE_AA,
        )
        if logo_width and logo_height:
            display[logo_y:logo_y + logo_height, logo_x:logo_x + logo_width] = siix_logo
        cv2.putText(
            display,
            header_title,
            (header_text_x + 2, header_text_y + 2),
            cv2.FONT_HERSHEY_DUPLEX,
            header_scale,
            (3, 6, 12),
            header_thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            header_title,
            (header_text_x, header_text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            header_scale,
            (245, 248, 255),
            header_thickness,
            cv2.LINE_AA,
        )

        if model is None:
            cv2.putText(
                display,
                "Model belum ada",
                (camera_x + 20, camera_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        tray_is_full = (
            model_selected_for_display
            and qty_for_display >= target_qty_for_display
        )
        if tray_is_full and alert_can_play and not alert_is_playing:
            alert_is_playing = send_mci_command(f"play {alert_alias} repeat")
            if not alert_is_playing:
                alert_can_play = False
        elif not tray_is_full and alert_is_playing:
            send_mci_command(f"stop {alert_alias}")
            alert_is_playing = False

        card_radius = max(18, round(screen_height * 0.035))
        card_glow_color = (120, 225, 35) if tray_is_full else (24, 70, 255)
        draw_rounded_rectangle(
            display,
            (qty_left - 6, qty_top - 6, qty_right + 6, qty_bottom + 6),
            (255, 170, 25),
            card_radius + 5,
        )
        draw_rounded_rectangle(
            display,
            (qty_left - 2, qty_top - 2, qty_right + 2, qty_bottom + 2),
            card_glow_color,
            card_radius + 2,
        )
        if tray_is_full:
            fill_rounded_gradient(
                display,
                (qty_left, qty_top, qty_right, qty_bottom),
                (24, 125, 38),
                (8, 60, 18),
                card_radius,
            )
        else:
            fill_rounded_gradient(
                display,
                (qty_left, qty_top, qty_right, qty_bottom),
                (25, 25, 205),
                (5, 5, 105),
                card_radius,
            )

        divider_x = qty_left + round(qty_panel_width * 0.20)
        cv2.line(
            display,
            (divider_x, qty_top + round((qty_bottom - qty_top) * 0.16)),
            (divider_x, qty_bottom - round((qty_bottom - qty_top) * 0.12)),
            (65, 85, 220),
            max(1, round(screen_height * 0.002)),
            cv2.LINE_AA,
        )
        icon_size = min(
            round(qty_panel_width * 0.15),
            round((qty_bottom - qty_top) * 0.42),
        )
        draw_pcb_tray_icon(
            display,
            (qty_left + round(qty_panel_width * 0.10), qty_top + (qty_bottom - qty_top) // 2),
            icon_size,
            (255, 255, 255),
            max(2, round(screen_height / 558 * 2)),
        )

        badge_height = max(40, round(screen_height * 0.095))
        badge_top = qty_bottom - round(screen_height * 0.16)
        badge_left = divider_x + round(qty_panel_width * 0.055)
        badge_right = qty_right - round(qty_panel_width * 0.055)
        badge_radius = max(12, badge_height // 2)
        badge_border = (95, 115, 255) if not tray_is_full else (100, 230, 90)
        badge_fill = (15, 15, 115) if not tray_is_full else (15, 65, 25)
        draw_rounded_rectangle(
            display,
            (badge_left, badge_top, badge_right, badge_top + badge_height),
            badge_border,
            badge_radius,
        )
        draw_rounded_rectangle(
            display,
            (
                badge_left + 3,
                badge_top + 3,
                badge_right - 3,
                badge_top + badge_height - 3,
            ),
            badge_fill,
            max(8, badge_radius - 3),
        )
        badge_icon_center = (
            badge_left + round(badge_height * 0.58),
            badge_top + badge_height // 2,
        )
        badge_icon_radius = max(12, round(badge_height * 0.30))
        cv2.circle(
            display,
            badge_icon_center,
            badge_icon_radius,
            (245, 248, 255),
            cv2.FILLED,
            cv2.LINE_AA,
        )
        if tray_is_full:
            check_left = (badge_icon_center[0] - badge_icon_radius // 2, badge_icon_center[1])
            check_mid = (badge_icon_center[0] - badge_icon_radius // 8, badge_icon_center[1] + badge_icon_radius // 3)
            check_right = (badge_icon_center[0] + badge_icon_radius // 2, badge_icon_center[1] - badge_icon_radius // 3)
            cv2.line(display, check_left, check_mid, (20, 125, 35), 3, cv2.LINE_AA)
            cv2.line(display, check_mid, check_right, (20, 125, 35), 3, cv2.LINE_AA)
            status_label = "COMPLETE (OK)"
        else:
            badge_symbol = "!" if model_selected_for_display else "?"
            exclamation_scale = badge_icon_radius / 17
            exclamation_thickness = max(2, round(badge_icon_radius / 6))
            (symbol_width, symbol_height), _ = cv2.getTextSize(
                badge_symbol,
                cv2.FONT_HERSHEY_DUPLEX,
                exclamation_scale,
                exclamation_thickness,
            )
            cv2.putText(
                display,
                badge_symbol,
                (
                    badge_icon_center[0] - symbol_width // 2,
                    badge_icon_center[1] + symbol_height // 2,
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                exclamation_scale,
                (25, 25, 180),
                exclamation_thickness,
                cv2.LINE_AA,
            )
            status_label = (
                "NOT COMPLETE (NG)"
                if model_selected_for_display
                else "SELECT MODEL"
            )

        status_text_left = badge_icon_center[0] + badge_icon_radius + round(badge_height * 0.28)
        status_text_right = badge_right - round(badge_height * 0.18)
        status_scale = screen_height / 558 * 0.78
        status_thickness = max(1, round(screen_height / 558 * 1.8))
        (status_text_width, status_text_height), _ = cv2.getTextSize(
            status_label,
            cv2.FONT_HERSHEY_DUPLEX,
            status_scale,
            status_thickness,
        )
        status_scale *= min(
            1.0,
            (status_text_right - status_text_left) / max(1, status_text_width),
        )
        status_thickness = max(1, round(status_thickness * min(1.0, status_scale / (screen_height / 558 * 0.78))))
        (status_text_width, status_text_height), _ = cv2.getTextSize(
            status_label,
            cv2.FONT_HERSHEY_DUPLEX,
            status_scale,
            status_thickness,
        )
        cv2.putText(
            display,
            status_label,
            (
                status_text_left + (status_text_right - status_text_left - status_text_width) // 2,
                badge_top + (badge_height + status_text_height) // 2,
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            status_scale,
            (245, 248, 255),
            status_thickness,
            cv2.LINE_AA,
        )

        qty_text = (
            f"{qty_for_display} / {target_qty_for_display}"
            if model_selected_for_display
            else "-- / --"
        )
        qty_scale = screen_height / 558 * 3.0
        qty_thickness = max(2, round(screen_height / 558 * 6))
        number_left = divider_x + round(qty_panel_width * 0.035)
        number_right = qty_right - round(qty_panel_width * 0.035)
        number_top = qty_top + round((qty_bottom - qty_top) * 0.10)
        number_bottom = badge_top - round(screen_height * 0.025)
        (qty_text_width, qty_text_height), _ = cv2.getTextSize(
            qty_text,
            cv2.FONT_HERSHEY_DUPLEX,
            qty_scale,
            qty_thickness,
        )
        max_text_width = (number_right - number_left) * 0.92
        max_text_height = number_bottom - number_top
        fit_scale = min(
            max_text_width / qty_text_width,
            max_text_height / qty_text_height,
            1.0,
        )
        qty_scale *= fit_scale
        qty_thickness = max(1, round(qty_thickness * fit_scale))
        (qty_text_width, qty_text_height), _ = cv2.getTextSize(
            qty_text,
            cv2.FONT_HERSHEY_DUPLEX,
            qty_scale,
            qty_thickness,
        )
        cv2.putText(
            display,
            qty_text,
            (
                number_left + (number_right - number_left - qty_text_width) // 2,
                number_top + (number_bottom - number_top + qty_text_height) // 2,
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            qty_scale,
            (255, 255, 255),
            qty_thickness,
            cv2.LINE_AA,
        )

        footer_radius = max(14, round(screen_height * 0.022))
        draw_rounded_rectangle(
            display,
            (qty_left - 5, model_footer_top - 4, qty_right + 5, model_footer_bottom + 4),
            (255, 150, 22),
            footer_radius + 4,
        )
        fill_rounded_gradient(
            display,
            (qty_left, model_footer_top, qty_right, model_footer_bottom),
            (34, 25, 13),
            (14, 12, 7),
            footer_radius,
        )

        control_center_y = model_control_y + model_control_height // 2
        monitor_size = min(
            round(model_control_height * 0.48),
            round(model_caption_area_width * 0.34),
        )
        monitor_center_x = model_label_left + round(model_caption_area_width * 0.24)
        draw_monitor_icon(
            display,
            (monitor_center_x, control_center_y),
            monitor_size,
            (245, 249, 255),
            max(2, round(screen_height / 558 * 1.5)),
        )
        model_caption = "Model"
        model_caption_scale = screen_height / 558 * 0.48
        model_caption_thickness = max(1, round(screen_height / 558 * 1.2))
        (_, model_caption_height), _ = cv2.getTextSize(
            model_caption,
            cv2.FONT_HERSHEY_DUPLEX,
            model_caption_scale,
            model_caption_thickness,
        )
        cv2.putText(
            display,
            model_caption,
            (
                model_label_left + round(model_caption_area_width * 0.43),
                control_center_y + model_caption_height // 2,
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            model_caption_scale,
            (245, 249, 255),
            model_caption_thickness,
            cv2.LINE_AA,
        )

        dropdown_radius = max(10, round(model_control_height * 0.18))
        draw_rounded_rectangle(
            display,
            (
                model_dropdown_x,
                model_control_y,
                model_dropdown_x + model_dropdown_width,
                model_control_y + model_control_height,
            ),
            (245, 249, 255),
            dropdown_radius,
        )
        fill_rounded_gradient(
            display,
            (
                model_dropdown_x + 2,
                model_control_y + 2,
                model_dropdown_x + model_dropdown_width - 2,
                model_control_y + model_control_height - 2,
            ),
            (249, 251, 255),
            (215, 228, 242),
            max(6, dropdown_radius - 2),
        )
        model_label = model_label_for_number(selected_model_number)
        if model_selection_locked:
            model_label += ""
        model_label_scale = screen_height / 558 * 0.60
        model_label_thickness = max(1, round(screen_height / 558 * 1.3))
        (model_text_width, model_label_height), _ = cv2.getTextSize(
            model_label,
            cv2.FONT_HERSHEY_DUPLEX,
            model_label_scale,
            model_label_thickness,
        )
        model_text_area_width = model_dropdown_width - max(
            38,
            round(model_dropdown_width * 0.13),
        )
        model_label_x = model_dropdown_x + (
            model_text_area_width - model_text_width
        ) // 2
        cv2.putText(
            display,
            model_label,
            (
                model_label_x,
                model_control_y
                + (model_control_height + model_label_height) // 2,
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            model_label_scale,
            (28, 32, 45),
            model_label_thickness,
            cv2.LINE_AA,
        )
        arrow_center_x = model_dropdown_x + model_dropdown_width - max(
            18, round(model_dropdown_width * 0.045)
        )
        arrow_center_y = model_control_y + model_control_height // 2
        if model_dropdown_open:
            arrow_points = np.array(
                [
                    [arrow_center_x - 9, arrow_center_y + 5],
                    [arrow_center_x + 9, arrow_center_y + 5],
                    [arrow_center_x, arrow_center_y - 6],
                ],
                dtype=np.int32,
            )
        else:
            arrow_points = np.array(
                [
                    [arrow_center_x - 9, arrow_center_y - 5],
                    [arrow_center_x + 9, arrow_center_y - 5],
                    [arrow_center_x, arrow_center_y + 6],
                ],
                dtype=np.int32,
            )
        cv2.fillPoly(display, [arrow_points], (30, 35, 50))

        change_radius = max(10, round(model_control_height * 0.18))
        draw_rounded_rectangle(
            display,
            (
                model_change_x - 2,
                model_control_y - 2,
                model_change_x + model_change_width + 2,
                model_control_y + model_control_height + 2,
            ),
            (35, 205, 255),
            change_radius + 2,
        )
        fill_rounded_gradient(
            display,
            (
                model_change_x,
                model_control_y,
                model_change_x + model_change_width,
                model_control_y + model_control_height,
            ),
            (35, 150, 255),
            (0, 86, 225),
            change_radius,
        )
        change_icon_radius = max(9, round(model_control_height * 0.20))
        change_icon_center = (
            model_change_x + round(model_change_width * 0.25),
            control_center_y,
        )
        draw_refresh_icon(
            display,
            change_icon_center,
            change_icon_radius,
            (255, 255, 255),
            max(2, round(screen_height / 558 * 2)),
        )
        change_label = "Change"
        change_scale = screen_height / 558 * 0.58
        change_thickness = max(1, round(screen_height / 558 * 1.3))
        (change_text_width, change_text_height), _ = cv2.getTextSize(
            change_label,
            cv2.FONT_HERSHEY_DUPLEX,
            change_scale,
            change_thickness,
        )
        change_text_left = model_change_x + round(model_change_width * 0.43)
        change_text_right = model_change_x + round(model_change_width * 0.94)
        if change_text_width > change_text_right - change_text_left:
            change_scale *= (change_text_right - change_text_left) / change_text_width
            change_thickness = max(1, round(change_thickness * 0.9))
            (change_text_width, change_text_height), _ = cv2.getTextSize(
                change_label,
                cv2.FONT_HERSHEY_DUPLEX,
                change_scale,
                change_thickness,
            )
        cv2.putText(
            display,
            change_label,
            (
                change_text_left + (change_text_right - change_text_left - change_text_width) // 2,
                control_center_y + change_text_height // 2,
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            change_scale,
            (255, 255, 255),
            change_thickness,
            cv2.LINE_AA,
        )

        if model_dropdown_open:
            for option_index, model_number in enumerate(MODEL_OPTIONS):
                option_top = model_menu_top + option_index * model_option_height
                option_bottom = option_top + model_option_height
                option_is_selected = model_number == selected_model_number
                option_radius = max(6, round(model_option_height * 0.20))
                option_border = (25, 170, 255) if option_is_selected else (190, 115, 25)
                option_fill_top = (245, 250, 255) if option_is_selected else (42, 29, 14)
                option_fill_bottom = (218, 233, 248) if option_is_selected else (22, 18, 10)
                draw_rounded_rectangle(
                    display,
                    (
                        model_dropdown_x,
                        option_top,
                        model_dropdown_x + model_dropdown_width,
                        option_bottom,
                    ),
                    option_border,
                    option_radius,
                )
                fill_rounded_gradient(
                    display,
                    (
                        model_dropdown_x + 2,
                        option_top + 2,
                        model_dropdown_x + model_dropdown_width - 2,
                        option_bottom - 2,
                    ),
                    option_fill_top,
                    option_fill_bottom,
                    max(4, option_radius - 2),
                )
                option_label = model_label_for_number(model_number)
                option_path = model_path_for_number(model_number)
                if option_path is None or not option_path.is_file():
                    option_label += " (file belum ada)"
                option_scale = screen_height / 558 * 0.48
                option_thickness = max(1, round(screen_height / 558))
                (option_text_width, option_text_height), _ = cv2.getTextSize(
                    option_label,
                    cv2.FONT_HERSHEY_DUPLEX,
                    option_scale,
                    option_thickness,
                )
                option_text_limit = model_dropdown_width - max(
                    16,
                    round(model_dropdown_width * 0.06),
                )
                if option_text_width > option_text_limit:
                    option_scale *= option_text_limit / option_text_width
                    option_thickness = max(1, round(option_thickness * 0.9))
                    (_, option_text_height), _ = cv2.getTextSize(
                        option_label,
                        cv2.FONT_HERSHEY_DUPLEX,
                        option_scale,
                        option_thickness,
                    )
                option_text_color = (25, 35, 50) if option_is_selected else (240, 246, 255)
                cv2.putText(
                    display,
                    option_label,
                    (
                        model_dropdown_x + max(8, round(model_dropdown_width * 0.03)),
                        option_top + (model_option_height + option_text_height) // 2,
                    ),
                    cv2.FONT_HERSHEY_DUPLEX,
                    option_scale,
                    option_text_color,
                    option_thickness,
                    cv2.LINE_AA,
                )

        if model_status_message and time.monotonic() < model_status_until:
            status_scale = screen_height / 558 * 0.38
            status_thickness = max(1, round(screen_height / 558))
            (status_width, _), _ = cv2.getTextSize(
                model_status_message,
                cv2.FONT_HERSHEY_DUPLEX,
                status_scale,
                status_thickness,
            )
            status_limit = qty_panel_width * 0.96
            if status_width > status_limit:
                status_scale *= status_limit / status_width
                status_thickness = max(1, round(status_thickness * 0.9))
                (status_width, _), _ = cv2.getTextSize(
                    model_status_message,
                    cv2.FONT_HERSHEY_DUPLEX,
                    status_scale,
                    status_thickness,
                )
            cv2.putText(
                display,
                model_status_message,
                (
                    qty_left + (qty_panel_width - status_width) // 2,
                    min(
                        screen_height - outer_margin - 4,
                        model_footer_bottom + round(screen_height * 0.032),
                    ),
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                status_scale,
                (255, 255, 255),
                status_thickness,
                cv2.LINE_AA,
            )

        if (
            model is not None
            and counting_enabled
            and now - last_inference_time >= INFERENCE_INTERVAL_SECONDS
        ):
            inference_frame = camera_frame.copy()
            with state_lock:
                generation = counting_generation
            try:
                inference_queue.put_nowait((inference_frame, generation))
            except queue.Full:
                try:
                    inference_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    inference_queue.put_nowait((inference_frame, generation))
                except queue.Full:
                    pass
            last_inference_time = now

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("s"), ord("S")):
            filename = datetime.now().strftime("tray_%Y%m%d_%H%M%S_%f.jpg")
            path = save_dir / filename
            if cv2.imwrite(str(path), camera_frame):
                print(f"Foto tersimpan: {path}")
            else:
                print("Gagal menyimpan foto.")
        elif key == ord("*"):
            break
finally:
    stop_event.set()
    inference_thread.join(timeout=2)
    if alert_is_open:
        if alert_is_playing:
            send_mci_command(f"stop {alert_alias}")
        send_mci_command(f"close {alert_alias}")
    if ng_alert_is_open:
        if ng_alert_is_playing:
            send_mci_command(f"stop {ng_alert_alias}")
        send_mci_command(f"close {ng_alert_alias}")
    camera.release()
    cv2.destroyAllWindows()

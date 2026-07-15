import argparse
import csv
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# Landmark IDs used by MediaPipe FaceMesh.
PUPIL_LEFT = 468
PUPIL_RIGHT = 473
LEFT_REF_X_LM = 55
LEFT_REF_Y_LM = 65
RIGHT_REF_X_LM = 285
RIGHT_REF_Y_LM = 295

LANDMARK_IDS = [
    33, 133, 362, 263,
    159, 145, 386, 374,
    55, 65, 285, 295,
    468, 473,
]

AREA_LABELS = {
    0: "LEFT UP",
    1: "RIGHT UP",
    2: "LEFT DOWN",
    3: "RIGHT DOWN",
}

DIRECTION_LABELS = AREA_LABELS
BAD_FISH_PENALTY = -10
BAD_FISH_SPAWN_SEC = 8.0
BAD_FISH_BOUNCE_JITTER_DEG = 38
BACKGROUND_IMAGE = None
BACKGROUND_CACHE = {}
BACKGROUND_ANIMATION_CACHE = {}
BACKGROUND_ANIMATION_FRAMES = 8
BACKGROUND_ANIMATION_FPS = 4.0
BACKGROUND_DARKEN_ALPHA = 0.82

FISH_TYPES = [
    {
        "name": "Goldfish",
        "score": 10,
        "is_bad": False,
        "speed": 185,
        "body_color": (55, 150, 255),
        "tail_color": (35, 105, 255),
        "accent_color": (255, 245, 215),
        "radius_scale": 1.00,
        "chance": 0.58,
    },
    {
        "name": "Red Goldfish",
        "score": 20,
        "is_bad": False,
        "speed": 245,
        "body_color": (45, 45, 235),
        "tail_color": (30, 30, 190),
        "accent_color": (255, 230, 230),
        "radius_scale": 0.95,
        "chance": 0.29,
    },
    {
        "name": "Rare Fish",
        "score": 50,
        "is_bad": False,
        "speed": 330,
        "body_color": (235, 215, 80),
        "tail_color": (245, 180, 35),
        "accent_color": (255, 255, 255),
        "radius_scale": 0.82,
        "chance": 0.13,
    },
    {
        "name": "Bad Fish",
        "score": BAD_FISH_PENALTY,
        "is_bad": True,
        "speed": 230,
        "body_color": (70, 70, 90),
        "tail_color": (35, 35, 55),
        "accent_color": (60, 60, 220),
        "radius_scale": 1.05,
        "chance": 0.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--camera-profile", choices=["webcam", "smartphone", "custom"], default="custom")

    parser.add_argument("--calib-sec", type=float, default=5.0)
    parser.add_argument("--ignore-sec", type=float, default=1.0)
    parser.add_argument("--game-sec", type=float, default=30.0)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--no-time-limit", action="store_true")
    parser.add_argument("--hit-effect-sec", type=float, default=0.65)
    parser.add_argument("--target-life-sec", type=float, default=5.0)
    parser.add_argument("--target-overlap-sec", type=float, default=3.0)
    parser.add_argument("--fish-spawn-sec", type=float, default=0.45)
    parser.add_argument("--max-fish", type=int, default=16)
    parser.add_argument("--poi-radius", type=int, default=118)
    parser.add_argument("--bonus-chance", type=float, default=0.20)
    parser.add_argument("--normal-score", type=int, default=1)
    parser.add_argument("--bonus-score", type=int, default=3)
    parser.add_argument("--enemy-radius", type=int, default=70)
    parser.add_argument("--enemy-color", choices=["cyan", "green", "yellow", "magenta", "red", "blue"], default="cyan")
    parser.add_argument("--dot-radius", type=int, default=24)
    parser.add_argument("--ranking-file", type=str, default="gaze_shooting_ranking.csv")
    parser.add_argument("--ranking-size", type=int, default=5)

    parser.add_argument("--dead-x", type=float, default=0.6)
    parser.add_argument("--dead-y", type=float, default=0.6)
    parser.add_argument("--x-scale", type=float, default=1.0)
    parser.add_argument("--y-scale", type=float, default=1.0)
    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--majority-window", type=int, default=1)

    parser.add_argument("--fullscreen", action="store_true")

    parser.add_argument("--disable-paper-preprocess", action="store_true")
    parser.add_argument("--no-zoom", action="store_true")
    parser.add_argument("--zoom-factor", type=float, default=3.33)
    parser.add_argument("--proc-width", type=int, default=1920)
    parser.add_argument("--proc-height", type=int, default=1080)
    parser.add_argument("--disable-gaussian", action="store_true")
    parser.add_argument("--gaussian-ksize", type=int, default=3)
    parser.add_argument("--paper-exact-crop", action="store_true")
    parser.add_argument("--calib-beta", type=float, default=0.27)
    parser.add_argument("--realtime-beta", type=float, default=0.20)

    parser.add_argument("--blink-ear-th", type=float, default=0.18)
    parser.add_argument("--post-blink-hold-frames", type=int, default=5)

    args = parser.parse_args()

    if args.camera_profile == "smartphone":
        args.width = 1280
        args.height = 720
        args.no_zoom = True
        args.disable_paper_preprocess = False
    elif args.camera_profile == "webcam":
        args.width = 640
        args.height = 480
        args.no_zoom = False
        args.disable_paper_preprocess = False

    return args


def scaled(value, base, current):
    return max(1, int(round(value * current / base)))


def put_text(img, text, y, color=(255, 255, 255), scale=1.0, thickness=2, x=40):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_centered_text(img, text, y, color=(255, 255, 255), scale=1.0, thickness=2):
    h, w = img.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (w - tw) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class LandmarkSmoother:
    def __init__(self):
        self.prev = {}

    def reset(self):
        self.prev.clear()

    def apply(self, lm, beta):
        beta = min(0.999, max(0.001, float(beta)))
        smoothed = {}
        for idx, (x, y) in lm.items():
            if idx in self.prev:
                px, py = self.prev[idx]
                sx = beta * x + (1.0 - beta) * px
                sy = beta * y + (1.0 - beta) * py
            else:
                sx, sy = x, y
            smoothed[idx] = (sx, sy)
            self.prev[idx] = (sx, sy)
        return smoothed


def get_preprocess_crop_rect(frame_shape, args):
    h, w = frame_shape[:2]

    if args.disable_paper_preprocess or args.no_zoom:
        return 0, 0, w, h

    if args.paper_exact_crop:
        x1 = int(round(w * 0.35))
        x2 = int(round(w * 0.65))
        y1 = int(round(h * 0.35))
        y2 = int(round(h * 0.65))
        return x1, y1, x2, y2

    z = max(1.0, float(args.zoom_factor))
    crop_w = max(2, int(round(w / z)))
    crop_h = max(2, int(round(h / z)))
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)
    return x1, y1, x2, y2


def paper_preprocess_frame(frame, args):
    if args.disable_paper_preprocess:
        proc = frame.copy()
    elif args.no_zoom:
        proc = cv2.resize(frame, (args.proc_width, args.proc_height), interpolation=cv2.INTER_CUBIC)
    else:
        x1, y1, x2, y2 = get_preprocess_crop_rect(frame.shape, args)
        crop = frame[y1:y2, x1:x2]
        proc = cv2.resize(crop, (args.proc_width, args.proc_height), interpolation=cv2.INTER_CUBIC)

    if (not args.disable_paper_preprocess) and (not args.disable_gaussian):
        k = int(args.gaussian_ksize)
        if k % 2 == 0:
            k += 1
        if k >= 3:
            proc = cv2.GaussianBlur(proc, (k, k), 0)
    return proc


def open_capture(args):
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(args.camera, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None and float(frame.mean()) >= 2.0:
                print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                return cap
            cv2.waitKey(30)

        cap.release()

    return None


def get_landmark_xy(landmarks, idx, image_w, image_h):
    if idx >= len(landmarks):
        return None, None
    lm = landmarks[idx]
    return lm.x * image_w, lm.y * image_h


def calc_ear(lm):
    left_w = max(1.0, abs(lm[133][0] - lm[33][0]))
    right_w = max(1.0, abs(lm[263][0] - lm[362][0]))
    left_h = abs(lm[145][1] - lm[159][1])
    right_h = abs(lm[374][1] - lm[386][1])
    return (left_h / left_w + right_h / right_w) / 2.0


def signed_brow_feature(lm):
    left_pupil_x, left_pupil_y = lm[PUPIL_LEFT]
    right_pupil_x, right_pupil_y = lm[PUPIL_RIGHT]
    return np.array([
        left_pupil_x - lm[LEFT_REF_X_LM][0],
        left_pupil_y - lm[LEFT_REF_Y_LM][1],
        right_pupil_x - lm[RIGHT_REF_X_LM][0],
        right_pupil_y - lm[RIGHT_REF_Y_LM][1],
    ], dtype=np.float64)


def extract_features(frame, results, smoother=None, smooth_beta=None):
    image_h, image_w = frame.shape[:2]
    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    if len(landmarks) < 478:
        return None, None

    lm = {}
    for idx in LANDMARK_IDS:
        x, y = get_landmark_xy(landmarks, idx, image_w, image_h)
        if x is None or y is None:
            return None, None
        lm[idx] = (x, y)

    raw_ear = calc_ear(lm)

    if smoother is not None and smooth_beta is not None:
        lm = smoother.apply(lm, smooth_beta)

    ear = calc_ear(lm)
    feature = signed_brow_feature(lm)

    left_eye_center_x = (lm[33][0] + lm[133][0]) / 2.0
    left_eye_center_y = (lm[33][1] + lm[133][1]) / 2.0
    right_eye_center_x = (lm[362][0] + lm[263][0]) / 2.0
    right_eye_center_y = (lm[362][1] + lm[263][1]) / 2.0

    info = {
        "eye_center_x": (left_eye_center_x + right_eye_center_x) / 2.0,
        "eye_center_y": (left_eye_center_y + right_eye_center_y) / 2.0,
        "inter_eye_dist": max(1.0, abs(right_eye_center_x - left_eye_center_x)),
        "ear": ear,
        "raw_ear": raw_ear,
    }

    for idx in LANDMARK_IDS:
        info[f"lm{idx}_x"] = lm[idx][0]
        info[f"lm{idx}_y"] = lm[idx][1]

    return feature, info


def process_frame(cap, face_mesh, args, smoother=None, smooth_beta=None):
    ret, raw_frame = cap.read()
    if not ret or raw_frame is None:
        return None, None, None

    proc_frame = paper_preprocess_frame(raw_frame, args)
    rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    feature, info = extract_features(proc_frame, results, smoother=smoother, smooth_beta=smooth_beta)
    return proc_frame, feature, info


def draw_landmark_view(frame, info):
    if frame is None:
        return None

    view = frame.copy()

    if info is None:
        put_text(view, "Face landmarks: not detected", 30, (0, 0, 255), scale=0.8, thickness=2)
        put_text(view, "Please face the camera", 65, (0, 0, 255), scale=0.7, thickness=2)
        return view

    for idx in LANDMARK_IDS:
        x = info.get(f"lm{idx}_x")
        y = info.get(f"lm{idx}_y")
        if x is None or y is None:
            continue

        if idx in (468, 473):
            color = (0, 0, 255)      # pupil
            radius = 5
        elif idx in (55, 65, 285, 295):
            color = (0, 255, 255)    # reference points
            radius = 5
        else:
            color = (0, 255, 0)      # eye landmarks
            radius = 4

        cv2.circle(view, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    put_text(view, "Face landmarks: detected", 30, (0, 255, 0), scale=0.8, thickness=2)
    put_text(view, "Red: pupil / Yellow: reference / Green: eye area", 65, (255, 255, 255), scale=0.6, thickness=2)

    return view


def draw_grid(screen):
    h, w = screen.shape[:2]
    line_th = scaled(2, 1080, h)
    cv2.line(screen, (w // 2, 0), (w // 2, h), (80, 80, 80), line_th)
    cv2.line(screen, (0, h // 2), (w, h // 2), (80, 80, 80), line_th)


def draw_wait_screen(w, h, radius):
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    draw_water_background(screen)
    draw_water_grass(screen)
    cx, cy = w // 2, h // 2 - scaled(40, 1080, h)
    r = scaled(radius, 1080, h)

    cv2.circle(screen, (cx, cy), r + scaled(14, 1080, h), (0, 255, 255), scaled(4, 1080, h), cv2.LINE_AA)
    cv2.circle(screen, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)

    draw_centered_text(screen, "KINGYO SUKUI GAZE GAME", scaled(120, 1080, h), color=(0, 255, 255), scale=1.5 * h / 1080, thickness=scaled(4, 1080, h))
    draw_centered_text(screen, "Look at the center dot, then press Enter", scaled(185, 1080, h), scale=0.95 * h / 1080, thickness=scaled(3, 1080, h))
    draw_centered_text(screen, "Catch fish by looking at a poi and blinking", scaled(245, 1080, h), color=(255, 255, 255), scale=0.85 * h / 1080, thickness=scaled(2, 1080, h))
    draw_centered_text(screen, "Press q to quit", h - scaled(80, 1080, h), scale=0.85 * h / 1080, thickness=scaled(2, 1080, h))
    return screen


def draw_calibration_screen(w, h, radius, remain_sec, elapsed, ignore_sec):
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    draw_water_background(screen)
    draw_water_grass(screen)
    cx, cy = w // 2, h // 2 - scaled(40, 1080, h)
    r = scaled(radius, 1080, h)

    cv2.circle(screen, (cx, cy), r + scaled(14, 1080, h), (0, 255, 255), scaled(4, 1080, h), cv2.LINE_AA)
    cv2.circle(screen, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)

    if elapsed < ignore_sec:
        text = "Get ready"
        color = (0, 255, 255)
    else:
        text = "Calibrating"
        color = (0, 255, 0)

    draw_centered_text(screen, text, scaled(120, 1080, h), color=color, scale=1.5 * h / 1080, thickness=scaled(4, 1080, h))
    draw_centered_text(screen, "Keep looking at the center dot", scaled(185, 1080, h), scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))
    draw_centered_text(screen, f"{remain_sec:.1f}s", cy + scaled(95, 1080, h), scale=1.3 * h / 1080, thickness=scaled(4, 1080, h))
    return screen


def calc_offset_xy(feature, center_feature, args):
    if feature is None or center_feature is None:
        return None, None
    diff = np.asarray(feature, dtype=np.float64) - np.asarray(center_feature, dtype=np.float64)
    dx = - (diff[0] + diff[2]) / 2.0
    dy = (diff[1] + diff[3]) / 2.0

    dx *= args.x_scale
    dy *= args.y_scale

    if args.invert_x:
        dx *= -1.0
    if args.invert_y:
        dy *= -1.0

    return float(dx), float(dy)


def classify_pure4(dx, dy, dead_x, dead_y):
    if dx is None or dy is None:
        return None
    if abs(dx) < dead_x and abs(dy) < dead_y:
        return None
    if dx < 0 and dy < 0:
        return 0
    if dx >= 0 and dy < 0:
        return 1
    if dx < 0 and dy >= 0:
        return 2
    return 3



def get_area_rect(w, h, area_id):
    col = area_id % 2
    row = area_id // 2
    x1 = col * w // 2
    y1 = row * h // 2
    x2 = (col + 1) * w // 2
    y2 = (row + 1) * h // 2
    return x1, y1, x2, y2


def get_area_center(w, h, area_id):
    x1, y1, x2, y2 = get_area_rect(w, h, area_id)
    return (x1 + x2) // 2, (y1 + y2) // 2


def get_enemy_color(name):
    colors = {
        "cyan": (255, 255, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "magenta": (255, 0, 255),
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
    }
    return colors.get(name, (255, 255, 0))


@dataclass
class Fish:
    fish_type: dict
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    wobble_seed: float

    @property
    def score(self):
        return int(self.fish_type["score"])

    @property
    def name(self):
        return self.fish_type["name"]

    @property
    def is_bad(self):
        return bool(self.fish_type.get("is_bad", False))

    def update(self, dt, w, h):
        self.x += self.vx * dt
        self.y += self.vy * dt
        bounced = False

        margin = max(self.radius + 10, scaled(42, 1080, h))
        if self.x < margin:
            self.x = margin
            self.vx = abs(self.vx)
            bounced = True
        elif self.x > w - margin:
            self.x = w - margin
            self.vx = -abs(self.vx)
            bounced = True

        top_margin = max(self.radius + 10, scaled(112, 1080, h))
        bottom_margin = max(self.radius + 10, scaled(70, 1080, h))
        if self.y < top_margin:
            self.y = top_margin
            self.vy = abs(self.vy)
            bounced = True
        elif self.y > h - bottom_margin:
            self.y = h - bottom_margin
            self.vy = -abs(self.vy)
            bounced = True

        if bounced and self.is_bad:
            self.randomize_bounce_angle()

    def randomize_bounce_angle(self):
        speed = max(1.0, math.hypot(self.vx, self.vy))
        angle = math.atan2(self.vy, self.vx)
        angle += math.radians(random.uniform(-BAD_FISH_BOUNCE_JITTER_DEG, BAD_FISH_BOUNCE_JITTER_DEG))
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed


def choose_fish_type(include_bad=False, bad_only=False):
    candidates = FISH_TYPES
    if bad_only:
        candidates = [fish_type for fish_type in FISH_TYPES if fish_type.get("is_bad", False)]
    elif not include_bad:
        candidates = [fish_type for fish_type in FISH_TYPES if not fish_type.get("is_bad", False)]

    r = random.random()
    acc = 0.0
    for fish_type in candidates:
        acc += fish_type["chance"]
        if r <= acc:
            return fish_type
    return candidates[-1]


def spawn_fish(w, h, bad_only=False):
    fish_type = choose_fish_type(bad_only=bad_only)
    cx = w * 0.5 + random.uniform(-w * 0.08, w * 0.08)
    cy = h * 0.52 + random.uniform(-h * 0.08, h * 0.08)
    angle = random.uniform(0.0, math.tau)
    speed = scaled(fish_type["speed"], 1080, h) * random.uniform(0.88, 1.15)
    radius = max(18, int(scaled(42, 1080, h) * fish_type["radius_scale"]))
    return Fish(
        fish_type=fish_type,
        x=cx,
        y=cy,
        vx=math.cos(angle) * speed,
        vy=math.sin(angle) * speed,
        radius=radius,
        wobble_seed=random.uniform(0.0, math.tau),
    )


def is_rare_fish(fish):
    name = str(fish.name if isinstance(fish, Fish) else fish.get("name", "")).lower()
    return "rare" in name or "special" in name


def spawn_initial_fishes(w, h, count):
    fishes = []
    max_initial_rare = max(1, count // 5)
    rare_count = 0
    attempts = 0
    while len(fishes) < count:
        fish = spawn_fish(w, h)
        attempts += 1
        if is_rare_fish(fish):
            if rare_count >= max_initial_rare and attempts < count * 8:
                continue
            rare_count += 1
        fishes.append(fish)
    return fishes


def update_fish(fishes, dt, w, h, args, last_spawn_time, last_bad_spawn_time):
    max_fish = max(1, int(args.max_fish))
    spawn_sec = max(0.25, float(args.fish_spawn_sec))
    now = time.perf_counter()

    for fish in fishes:
        fish.update(dt, w, h)

    if len(fishes) < max_fish and now - last_spawn_time >= spawn_sec:
        fishes.append(spawn_fish(w, h))
        last_spawn_time = now

    if now - last_bad_spawn_time >= BAD_FISH_SPAWN_SEC:
        fishes.append(spawn_fish(w, h, bad_only=True))
        last_bad_spawn_time = now

    if not fishes:
        fishes.append(spawn_fish(w, h))
        last_spawn_time = now

    return last_spawn_time, last_bad_spawn_time


def get_poi_center(w, h, direction):
    return get_area_center(w, h, direction)


def get_poi_radius(h, args):
    return scaled(args.poi_radius, 1080, h)


def fish_in_poi(fish, w, h, direction, args):
    px, py = get_poi_center(w, h, direction)
    catch_radius = get_poi_radius(h, args) + fish.radius * 0.45
    return math.hypot(fish.x - px, fish.y - py) <= catch_radius


def load_background_image():
    global BACKGROUND_IMAGE
    if BACKGROUND_IMAGE is not None:
        return BACKGROUND_IMAGE

    img_dir = Path(__file__).resolve().parent.parent / "img"
    image = cv2.imread(str(img_dir / "back_water.png"), cv2.IMREAD_COLOR)
    BACKGROUND_IMAGE = image
    return BACKGROUND_IMAGE


def draw_background_image(screen):
    background = load_background_image()
    if background is None:
        return False

    h, w = screen.shape[:2]
    cache_key = (w, h)
    frames = BACKGROUND_ANIMATION_CACHE.get(cache_key)
    if frames is None:
        resized = BACKGROUND_CACHE.get(cache_key)
        if resized is None:
            resized = cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)
            resized = cv2.convertScaleAbs(resized, alpha=BACKGROUND_DARKEN_ALPHA, beta=0)
            BACKGROUND_CACHE[cache_key] = resized

        base_x, base_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        y_norm = base_y / max(1.0, float(h - 1))
        x_norm = base_x / max(1.0, float(w - 1))
        amp_x = max(0.5, w * 0.0022)
        amp_y = max(0.25, h * 0.0010)
        frames = []
        for i in range(BACKGROUND_ANIMATION_FRAMES):
            phase = i / float(BACKGROUND_ANIMATION_FRAMES) * math.tau
            wave_x = np.sin(y_norm * math.tau * 2.0 + phase) * amp_x
            wave_x += np.sin(y_norm * math.tau * 4.0 - phase * 1.2) * amp_x * 0.18
            wave_y = np.sin(x_norm * math.tau * 1.4 - phase * 0.6) * amp_y
            map_x = (base_x + wave_x).astype(np.float32)
            map_y = (base_y + wave_y).astype(np.float32)
            frames.append(cv2.remap(
                resized,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ))
        BACKGROUND_ANIMATION_CACHE[cache_key] = frames

    frame_index = int(time.perf_counter() * BACKGROUND_ANIMATION_FPS) % len(frames)
    screen[:, :] = frames[frame_index]
    return True


def draw_water_background(screen):
    if draw_background_image(screen):
        return

    h, w = screen.shape[:2]
    top = np.array([105, 70, 28], dtype=np.float32)
    bottom = np.array([185, 135, 55], dtype=np.float32)
    for y in range(h):
        t = y / max(1, h - 1)
        color = (top * (1.0 - t) + bottom * t).astype(np.uint8)
        screen[y, :] = color

    wave_color = (215, 190, 120)
    for i, y in enumerate(range(scaled(120, 1080, h), h, scaled(130, 1080, h))):
        phase = i * scaled(72, 1920, w)
        pts = []
        for x in range(-scaled(80, 1920, w), w + scaled(80, 1920, w), scaled(36, 1920, w)):
            yy = int(y + math.sin((x + phase) / max(1, scaled(90, 1920, w))) * scaled(12, 1080, h))
            pts.append((x, yy))
        cv2.polylines(screen, [np.array(pts, dtype=np.int32)], False, wave_color, scaled(2, 1080, h), cv2.LINE_AA)

    cv2.circle(screen, (w // 2, h // 2), scaled(92, 1080, h), (210, 175, 105), scaled(3, 1080, h), cv2.LINE_AA)


def draw_poi(screen, direction, args, selected=False, blink=False, broken=False):
    h, w = screen.shape[:2]
    cx, cy = get_poi_center(w, h, direction)
    r = get_poi_radius(h, args)
    th = scaled(5, 1080, h)
    color = (255, 255, 255)
    rim = (0, 255, 255) if selected else (235, 235, 220)
    if broken:
        color = (120, 135, 135)
        rim = (95, 95, 95)
    if blink and selected and not broken:
        rim = (255, 0, 255)
    if selected:
        r = int(r * 1.13)
        th = scaled(9, 1080, h)

    cv2.circle(screen, (cx, cy), r, rim, th, cv2.LINE_AA)
    cv2.circle(screen, (cx, cy), int(r * 0.78), color, scaled(2, 1080, h), cv2.LINE_AA)
    for i in range(-2, 3):
        offset = i * r // 5
        cv2.line(screen, (cx - r + scaled(16, 1080, h), cy + offset), (cx + r - scaled(16, 1080, h), cy + offset), (220, 235, 235), 1, cv2.LINE_AA)
        cv2.line(screen, (cx + offset, cy - r + scaled(16, 1080, h)), (cx + offset, cy + r - scaled(16, 1080, h)), (220, 235, 235), 1, cv2.LINE_AA)

    if broken:
        x_size = int(r * 0.72)
        cv2.line(screen, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), (40, 40, 230), scaled(8, 1080, h), cv2.LINE_AA)
        cv2.line(screen, (cx + x_size, cy - x_size), (cx - x_size, cy + x_size), (40, 40, 230), scaled(8, 1080, h), cv2.LINE_AA)
        (bw, _), _ = cv2.getTextSize("BROKEN", cv2.FONT_HERSHEY_SIMPLEX, 0.72 * h / 1080, scaled(3, 1080, h))
        cv2.putText(
            screen,
            "BROKEN",
            (cx - bw // 2, cy + scaled(12, 1080, h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72 * h / 1080,
            (40, 40, 230),
            scaled(3, 1080, h),
            cv2.LINE_AA,
        )

    label = DIRECTION_LABELS[direction]
    (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * h / 1080, scaled(3, 1080, h))
    cv2.putText(
        screen,
        label,
        (cx - tw // 2, cy + r + th_text + scaled(22, 1080, h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9 * h / 1080,
        rim,
        scaled(3, 1080, h),
        cv2.LINE_AA,
    )


def draw_all_pois(screen, selected_direction, args, blink=False, broken_pois=None):
    broken_pois = broken_pois or set()
    for direction in (0, 1, 2, 3):
        draw_poi(
            screen,
            direction,
            args,
            selected=(direction == selected_direction),
            blink=blink,
            broken=direction in broken_pois,
        )


def draw_fish(screen, fish, now):
    h, _ = screen.shape[:2]
    angle = math.degrees(math.atan2(fish.vy, fish.vx))
    wiggle = math.sin(now * 8.0 + fish.wobble_seed) * 8.0
    body_r = fish.radius
    cx, cy = int(fish.x), int(fish.y)
    body_color = fish.fish_type["body_color"]
    tail_color = fish.fish_type["tail_color"]
    accent_color = fish.fish_type["accent_color"]

    theta = math.atan2(fish.vy, fish.vx)
    back_x = fish.x - math.cos(theta) * body_r * 1.18
    back_y = fish.y - math.sin(theta) * body_r * 1.18
    side_x = math.cos(theta + math.pi / 2.0)
    side_y = math.sin(theta + math.pi / 2.0)
    tail_len = body_r * 1.08
    tail_w = body_r * 0.78
    tail_tip = (int(back_x - math.cos(theta) * tail_len), int(back_y - math.sin(theta) * tail_len))
    tail_pts = np.array([
        tail_tip,
        (int(back_x + side_x * tail_w), int(back_y + side_y * tail_w)),
        (int(back_x - side_x * tail_w), int(back_y - side_y * tail_w)),
    ], dtype=np.int32)
    cv2.fillConvexPoly(screen, tail_pts, tail_color, cv2.LINE_AA)

    cv2.ellipse(
        screen,
        (cx, cy),
        (int(body_r * 1.15), int(body_r * 0.68)),
        angle + wiggle,
        0,
        360,
        body_color,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        screen,
        (cx, cy),
        (int(body_r * 0.72), int(body_r * 0.38)),
        angle + wiggle,
        0,
        360,
        accent_color,
        scaled(3, 1080, h),
        cv2.LINE_AA,
    )

    eye_x = int(fish.x + math.cos(theta) * body_r * 0.62 + side_x * body_r * 0.25)
    eye_y = int(fish.y + math.sin(theta) * body_r * 0.62 + side_y * body_r * 0.25)
    cv2.circle(screen, (eye_x, eye_y), max(3, body_r // 8), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(screen, (eye_x, eye_y), max(2, body_r // 14), (20, 20, 20), -1, cv2.LINE_AA)

    if fish.is_bad:
        mark_scale = 0.9 * h / 1080
        mark_th = scaled(3, 1080, h)
        cv2.putText(
            screen,
            "!!",
            (cx - body_r // 2, cy - body_r - scaled(18, 1080, h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            mark_scale,
            (40, 40, 255),
            mark_th,
            cv2.LINE_AA,
        )
        brow_len = max(8, body_r // 2)
        cv2.line(
            screen,
            (eye_x - brow_len, eye_y - brow_len // 2),
            (eye_x + brow_len // 2, eye_y - brow_len),
            (20, 20, 20),
            scaled(4, 1080, h),
            cv2.LINE_AA,
        )


def add_catch_effect(effects, fish, args):
    now = time.perf_counter()
    effects.append({
        "x": fish.x,
        "y": fish.y,
        "text": f"GET! +{fish.score}",
        "color": (0, 255, 255),
        "start": now,
        "until": now + args.hit_effect_sec,
    })


def add_miss_effect(effects, fish, args):
    now = time.perf_counter()
    effects.append({
        "x": fish.x,
        "y": fish.y,
        "text": f"MISS! {BAD_FISH_PENALTY}",
        "color": (40, 40, 255),
        "start": now,
        "until": now + args.hit_effect_sec,
    })
    effects.append({
        "x": fish.x,
        "y": fish.y + scaled(58, 1080, args.screen_height),
        "text": "POI BROKEN!",
        "color": (40, 40, 255),
        "start": now,
        "until": now + args.hit_effect_sec,
    })


def draw_effects(screen, effects):
    h, _ = screen.shape[:2]
    now = time.perf_counter()
    effects[:] = [effect for effect in effects if now < effect["until"]]
    for effect in effects:
        life = max(0.001, effect["until"] - effect["start"])
        t = (now - effect["start"]) / life
        y = int(effect["y"] - scaled(80, 1080, h) * t)
        scale = (1.3 + 0.4 * (1.0 - t)) * h / 1080
        cv2.putText(
            screen,
            effect["text"],
            (int(effect["x"] - scaled(95, 1920, screen.shape[1])), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            effect["color"],
            scaled(4, 1080, h),
            cv2.LINE_AA,
        )
        cv2.circle(screen, (int(effect["x"]), int(effect["y"])), int(scaled(70, 1080, h) * (1.0 + t)), effect["color"], scaled(3, 1080, h), cv2.LINE_AA)


def draw_game_screen(w, h, fishes, selected_direction, score, remain_sec, args,
                     blink=False, head_warning=False, effects=None, broken_pois=None):
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    draw_water_background(screen)
    draw_all_pois(screen, selected_direction, args, blink=blink, broken_pois=broken_pois)

    now = time.perf_counter()
    for fish in fishes:
        draw_fish(screen, fish, now)

    draw_water_grass(screen)

    if effects is not None:
        draw_effects(screen, effects)

    panel_h = scaled(86, 1080, h)
    cv2.rectangle(screen, (0, 0), (w, panel_h), (45, 70, 55), -1)
    cv2.line(screen, (0, panel_h), (w, panel_h), (0, 220, 220), scaled(3, 1080, h), cv2.LINE_AA)
    put_text(screen, f"SCORE: {score}", scaled(55, 1080, h), color=(255, 255, 255), scale=1.15 * h / 1080, thickness=scaled(3, 1080, h), x=scaled(35, 1920, w))

    direction_text = "SELECT: --" if selected_direction is None else f"SELECT: {DIRECTION_LABELS[selected_direction]}"
    if selected_direction is not None and broken_pois and selected_direction in broken_pois:
        direction_text += " (BROKEN)"
    draw_centered_text(screen, direction_text, scaled(55, 1080, h), color=(0, 255, 255), scale=0.95 * h / 1080, thickness=scaled(3, 1080, h))

    time_text = f"TIME: {remain_sec:.1f}"
    (tw, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1.15 * h / 1080, scaled(3, 1080, h))
    cv2.putText(screen, time_text, (w - tw - scaled(35, 1920, w), scaled(55, 1080, h)), cv2.FONT_HERSHEY_SIMPLEX, 1.15 * h / 1080, (255, 255, 255), scaled(3, 1080, h), cv2.LINE_AA)

    if blink:
        draw_centered_text(screen, "SCOOP!", h - scaled(95, 1080, h), color=(255, 0, 255), scale=1.15 * h / 1080, thickness=scaled(4, 1080, h))
    elif head_warning:
        draw_centered_text(screen, "Please keep your head still", h - scaled(95, 1080, h), color=(0, 0, 255), scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))

    put_text(screen, "Look LEFT UP / RIGHT UP / LEFT DOWN / RIGHT DOWN, blink when a fish enters the poi.  Press q to quit", h - scaled(35, 1080, h), color=(255, 255, 255), scale=0.66 * h / 1080, thickness=scaled(2, 1080, h))
    return screen



def get_ranking_path(args):
    path = Path(args.ranking_file)
    if path.is_absolute():
        return path
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    return base_dir / path


def load_rankings(args):
    path = get_ranking_path(args)
    if not path.exists():
        return []

    rankings = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    score = int(row.get("score", 0))
                except ValueError:
                    continue
                rankings.append({
                    "score": score,
                    "played_at": row.get("played_at", ""),
                })
    except OSError:
        return []

    rankings.sort(key=lambda x: x["score"], reverse=True)
    return rankings[:max(1, int(args.ranking_size))]


def save_rankings(args, rankings):
    path = get_ranking_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)

    rankings = sorted(rankings, key=lambda x: x["score"], reverse=True)
    rankings = rankings[:max(1, int(args.ranking_size))]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "score", "played_at"])
        writer.writeheader()
        for rank, row in enumerate(rankings, start=1):
            writer.writerow({
                "rank": rank,
                "score": int(row["score"]),
                "played_at": row.get("played_at", ""),
            })


def add_ranking_score(score, args):
    rankings = load_rankings(args)
    rankings.append({
        "score": int(score),
        "played_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    rankings.sort(key=lambda x: x["score"], reverse=True)
    rankings = rankings[:max(1, int(args.ranking_size))]
    try:
        save_rankings(args, rankings)
    except OSError as e:
        print(f"[WARN] Ranking could not be saved: {e}")
    return rankings

def draw_time_up_screen(w, h, score, rankings=None):
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    draw_water_background(screen)
    draw_water_grass(screen)
    draw_centered_text(screen, "TIME UP!", scaled(210, 1080, h), color=(0, 255, 255), scale=2.1 * h / 1080, thickness=scaled(6, 1080, h))
    draw_centered_text(screen, f"FINAL SCORE: {score}", scaled(325, 1080, h), scale=1.7 * h / 1080, thickness=scaled(5, 1080, h))

    draw_centered_text(screen, "RANKING", scaled(455, 1080, h), color=(255, 255, 0), scale=1.25 * h / 1080, thickness=scaled(4, 1080, h))

    if rankings:
        start_y = scaled(525, 1080, h)
        row_gap = scaled(55, 1080, h)
        for i, row in enumerate(rankings[:5], start=1):
            rank_text = f"{i:>2}.  SCORE {int(row['score'])}"
            draw_centered_text(screen, rank_text, start_y + (i - 1) * row_gap, scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))
    else:
        draw_centered_text(screen, "No ranking yet", scaled(545, 1080, h), scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))

    draw_centered_text(screen, "Press Enter to retry", h - scaled(145, 1080, h), scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))
    draw_centered_text(screen, "Press q to quit", h - scaled(85, 1080, h), scale=0.85 * h / 1080, thickness=scaled(2, 1080, h))
    return screen

def run_center_calibration(cap, face_mesh, args, smoother):
    smoother.reset()
    samples = []
    start = time.perf_counter()

    while True:
        now = time.perf_counter()
        elapsed = now - start
        if elapsed >= args.calib_sec:
            break

        frame, feature, info = process_frame(cap, face_mesh, args, smoother, args.calib_beta)
        if frame is None:
            continue
        landmark_view = draw_landmark_view(frame, info)
        if landmark_view is not None:
            cv2.imshow("camera_landmarks", landmark_view)
        blink_rejected = bool(info is not None and min(info.get("raw_ear", info["ear"]), info["ear"]) < args.blink_ear_th)
        if elapsed >= args.ignore_sec and feature is not None and info is not None and not blink_rejected:
            samples.append(feature)

        screen = draw_calibration_screen(
            args.screen_width,
            args.screen_height,
            args.dot_radius,
            max(0.0, args.calib_sec - elapsed),
            elapsed,
            args.ignore_sec,
        )
        cv2.imshow("gaze_shooting", screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None, None, True

    if not samples:
        return None, None, False

    center_feature = np.median(np.asarray(samples, dtype=np.float64), axis=0)
    _, _, center_info = process_frame(cap, face_mesh, args, smoother, args.calib_beta)
    return center_feature, center_info, False


def main():
    args = parse_args()

    cap = open_capture(args)
    if cap is None:
        print("[ERROR] Camera could not be opened.")
        return

    cv2.namedWindow("gaze_shooting", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gaze_shooting", args.screen_width, args.screen_height)
    cv2.namedWindow("camera_landmarks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera_landmarks", 640, 480)
    if args.fullscreen:
        cv2.setWindowProperty("gaze_shooting", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mp_face_mesh = mp.solutions.face_mesh
    calib_smoother = LandmarkSmoother()
    realtime_smoother = LandmarkSmoother()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            baseline_info = None
            baseline_feature = None
            calib_smoother.reset()

            while True:
                frame, feature, info = process_frame(cap, face_mesh, args, calib_smoother, args.calib_beta)

                landmark_view = draw_landmark_view(frame, info)
                if landmark_view is not None:
                    cv2.imshow("camera_landmarks", landmark_view)
                wait_screen = draw_wait_screen(args.screen_width, args.screen_height, args.dot_radius)
                cv2.imshow("gaze_shooting", wait_screen)

                if info is not None and feature is not None:
                    baseline_info = info
                    baseline_feature = feature

                key = cv2.waitKey(1) & 0xFF
                if key in (13, 10) and baseline_info is not None and baseline_feature is not None:
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            if args.skip_calibration:
                center_feature, center_info, quit_requested = baseline_feature, baseline_info, False
            else:
                center_feature, center_info, quit_requested = run_center_calibration(
                    cap, face_mesh, args, calib_smoother
                )
                if quit_requested:
                    break
            if center_feature is None:
                continue
            if center_info is None:
                center_info = baseline_info

            try:
                cv2.destroyWindow("camera_landmarks")
            except cv2.error:
                pass

            base_eye_x = center_info["eye_center_x"]
            base_eye_y = center_info["eye_center_y"]
            base_eye_dist = center_info["inter_eye_dist"]

            realtime_smoother.reset()
            pred_history = deque(maxlen=max(1, args.majority_window))
            last_valid_area = None
            post_blink_count = 0
            prev_blink = False
            score = 0
            initial_fish_count = min(max(4, int(args.max_fish) // 3), int(args.max_fish))
            fishes = spawn_initial_fishes(args.screen_width, args.screen_height, initial_fish_count)
            effects = []
            broken_pois = set()
            last_spawn_time = time.perf_counter()
            last_bad_spawn_time = time.perf_counter()
            prev_frame_time = time.perf_counter()
            game_start_time = time.perf_counter()
            
            fps_start = time.perf_counter()
            fps_frames = 0
            fps = 0.0

            while True:
                game_elapsed = time.perf_counter() - game_start_time
                game_remain = game_elapsed if args.no_time_limit else max(0.0, args.game_sec - game_elapsed)
                if (not args.no_time_limit) and game_elapsed >= args.game_sec:
                    break

                now_frame = time.perf_counter()
                dt = min(0.05, max(0.001, now_frame - prev_frame_time))
                prev_frame_time = now_frame
                last_spawn_time, last_bad_spawn_time = update_fish(
                    fishes,
                    dt,
                    args.screen_width,
                    args.screen_height,
                    args,
                    last_spawn_time,
                    last_bad_spawn_time,
                )

                frame, feature, info = process_frame(cap, face_mesh, args, realtime_smoother, args.realtime_beta)

                dx, dy = calc_offset_xy(feature, center_feature, args)
                head_warning = False
                blink_now = False

                if feature is not None and info is not None:
                    head_dx = abs(info["eye_center_x"] - base_eye_x)
                    head_dy = abs(info["eye_center_y"] - base_eye_y)
                    scale_change = abs(info["inter_eye_dist"] - base_eye_dist) / max(1.0, base_eye_dist)
                    head_warning = head_dx > 45 or head_dy > 35 or scale_change > 0.12

                    blink_metric = min(info.get("raw_ear", info["ear"]), info["ear"])
                    blink_now = blink_metric < args.blink_ear_th

                    blink_trigger = blink_now and not prev_blink

                    if blink_trigger and last_valid_area is not None and last_valid_area not in broken_pois:
                        hit_fishes = [
                            fish
                            for fish in fishes
                            if fish_in_poi(fish, args.screen_width, args.screen_height, last_valid_area, args)
                        ]
                        if hit_fishes:
                            caught_bad = False
                            for hit_fish in hit_fishes:
                                if hit_fish.is_bad:
                                    score += BAD_FISH_PENALTY
                                    caught_bad = True
                                    add_miss_effect(effects, hit_fish, args)
                                else:
                                    score += hit_fish.score
                                    add_catch_effect(effects, hit_fish, args)
                            if caught_bad:
                                broken_pois.add(last_valid_area)

                            hit_fish_ids = {id(fish) for fish in hit_fishes}
                            fishes[:] = [fish for fish in fishes if id(fish) not in hit_fish_ids]
                            while len(fishes) < max(2, int(args.max_fish) // 2):
                                fishes.append(spawn_fish(args.screen_width, args.screen_height))
                                last_spawn_time = time.perf_counter()

                    if blink_now:
                        post_blink_count = args.post_blink_hold_frames
                    elif post_blink_count > 0:
                        post_blink_count -= 1
                    else:
                        raw_direction = classify_pure4(dx, dy, args.dead_x, args.dead_y)
                        if raw_direction is not None:
                            pred_history.append(raw_direction)
                            last_valid_area = Counter(pred_history).most_common(1)[0][0]

                    prev_blink = blink_now
                else:
                    prev_blink = False

                screen = draw_game_screen(
                    args.screen_width,
                    args.screen_height,
                    fishes,
                    last_valid_area,
                    score,
                    game_remain,
                    args,
                    blink=blink_now,
                    head_warning=head_warning,
                    effects=effects,
                    broken_pois=broken_pois,
                )

                fps_frames += 1
                now = time.perf_counter()

                if now - fps_start >= 1.0:
                    fps = fps_frames / (now - fps_start)
                    print(f"FPS = {fps:.2f}")
                    fps_start = now
                    fps_frames = 0

                cv2.imshow("gaze_shooting", screen)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            rankings = add_ranking_score(score, args)

            while True:
                cv2.imshow("gaze_shooting", draw_time_up_screen(args.screen_width, args.screen_height, score, rankings))
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 10):
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()


_draw_fish_shape_fallback = draw_fish
FISH_SPRITES = None
FISH_RESIZED_SPRITE_CACHE = {}
FISH_WARP_GRID_CACHE = {}
FISH_RENDER_CACHE = {}
FISH_RENDER_CACHE_MAX = 160
FISH_ANIMATION_FPS = 8.0
FISH_ANGLE_BUCKET_DEG = 8.0
FISH_WIGGLE_SPEED = 2.35
FISH_WIGGLE_SPEED_BY_MOVE = 0.18
FISH_WIGGLE_SPEED_MAX_BONUS = 1.8
FISH_WIGGLE_BODY_WAVE = 1.05
FISH_COLOR_SATURATION_SCALE = 1.12
FISH_COLOR_VALUE_SCALE = 0.96
FISH_SHADOW_ENABLED = True
FISH_SHADOW_ANIMATION_DIVISOR = 2
FISH_SHADOW_ALPHA_SCALE = 0.58
GOLD_FISH_SHADOW_SCALE = 0.84
GOLD_FISH_OUTLINE_ALPHA = 135
GOLD_FISH_OUTLINE_SIZE = 5
WATER_GRASS_SPRITES = None
WATER_GRASS_RESIZED_CACHE = {}
WATER_GRASS_DARKEN_ALPHA = 0.86
WATER_GRASS_SWAY_FPS = 1.2
WATER_GRASS_SWAY_X_RATIO = 0.0015
WATER_GRASS_SWAY_Y_RATIO = 0.0006


def load_fish_sprites():
    global FISH_SPRITES
    if FISH_SPRITES is not None:
        return FISH_SPRITES

    import os

    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "img"))
    sprite_files = {
        "normal": "orange_fish.png",
        "goldfish": "orange_fish.png",
        "red": "red_fish.png",
        "rare": "gold_fish_outlined.png",
        "bad": "black_fish.png",
        "bad_mark": "black_fish_mark.png",
        "normal_shadow": "orange_fish_shadow.png",
        "red_shadow": "red_fish_shadow.png",
        "rare_shadow": "gold_fish_shadow.png",
        "bad_shadow": "black_fish_shadow.png",
    }
    sprites = {}
    for key, filename in sprite_files.items():
        path = os.path.join(img_dir, filename)
        sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if sprite is None and key == "rare":
            sprite = cv2.imread(os.path.join(img_dir, "gold_fish.png"), cv2.IMREAD_UNCHANGED)
        if sprite is None:
            continue
        if sprite.ndim == 2:
            sprite = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGRA)
        elif sprite.shape[2] == 3:
            alpha = np.full(sprite.shape[:2] + (1,), 255, dtype=np.uint8)
            sprite = np.concatenate([sprite, alpha], axis=2)
        if key.endswith("_shadow"):
            sprite = sprite.copy()
            sprite[:, :, 3] = np.clip(sprite[:, :, 3].astype(np.float32) * FISH_SHADOW_ALPHA_SCALE, 0, 255).astype(np.uint8)
        elif key not in ("bad_mark",):
            sprite = sprite.copy()
            hsv = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * FISH_COLOR_SATURATION_SCALE, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * FISH_COLOR_VALUE_SCALE, 0, 255)
            sprite[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        sprites[key] = sprite

    if "bad" in sprites and "bad_mark" in sprites:
        bad = sprites["bad"].copy()
        mark = sprites["bad_mark"]
        bad_h, bad_w = bad.shape[:2]
        mark_h, mark_w = mark.shape[:2]
        target_mark_h = max(10, int(bad_h * 0.34))
        target_mark_w = max(10, int(target_mark_h * mark_w / max(1, mark_h)))
        mark = cv2.resize(mark, (target_mark_w, target_mark_h), interpolation=cv2.INTER_AREA)

        y1 = max(0, int(bad_h * 0.04))
        x1 = min(max(0, bad_w - target_mark_w - int(bad_w * 0.04)), bad_w - 1)
        x2 = min(bad_w, x1 + target_mark_w)
        y2 = min(bad_h, y1 + target_mark_h)
        mark = mark[: y2 - y1, : x2 - x1]

        alpha = mark[:, :, 3:4].astype(np.uint16)
        bad_roi = bad[y1:y2, x1:x2].astype(np.uint16)
        mark_rgb = mark[:, :, :3].astype(np.uint16)
        bad[y1:y2, x1:x2, :3] = ((mark_rgb * alpha + bad_roi[:, :, :3] * (255 - alpha) + 127) // 255).astype(np.uint8)
        bad[y1:y2, x1:x2, 3:4] = np.maximum(bad_roi[:, :, 3:4], alpha).astype(np.uint8)
        sprites["bad"] = bad

    FISH_SPRITES = sprites
    return FISH_SPRITES


def _fish_value(fish, name, default=None):
    if isinstance(fish, dict):
        return fish.get(name, default)
    return getattr(fish, name, default)


def _fish_type_text(fish):
    candidates = [
        _fish_value(fish, "fish_type"),
        _fish_value(fish, "type_info"),
        _fish_value(fish, "type_name"),
        _fish_value(fish, "type"),
        _fish_value(fish, "kind"),
        _fish_value(fish, "name"),
        _fish_value(fish, "label"),
        _fish_value(fish, "color"),
    ]
    if _fish_value(fish, "is_bad", False):
        return "bad"
    if _fish_value(fish, "is_rare", False):
        return "rare"
    for value in candidates:
        if isinstance(value, dict):
            value = value.get("name") or value.get("type") or value.get("kind") or value.get("color")
        if value:
            return str(value).lower()
    return "normal"


def _fish_sprite_key(fish):
    text = _fish_type_text(fish)
    if "bad" in text or "black" in text:
        return "bad"
    if "rare" in text or "special" in text:
        return "rare"
    if "red" in text:
        return "red"
    return "normal"


def _fish_shadow_key(sprite_key):
    return f"{sprite_key}_shadow"


def prune_fish_render_cache(current_frame_bucket):
    if len(FISH_RENDER_CACHE) > FISH_RENDER_CACHE_MAX:
        FISH_RENDER_CACHE.clear()


def warp_fish_sprite(sprite, fish, now, cache_sprite_key=None, size_multiplier=1.0):
    radius = max(8, int(_fish_value(fish, "radius", 24)))
    src_h, src_w = sprite.shape[:2]
    sprite_key = cache_sprite_key or _fish_sprite_key(fish)
    size_scale = 4.6 if str(sprite_key).startswith("bad") else 3.8
    target_h = max(14, int(radius * size_scale * size_multiplier))
    target_w = max(8, int(target_h * src_w / max(1, src_h)))

    resized_key = (sprite_key, target_w, target_h)
    resized = FISH_RESIZED_SPRITE_CACHE.get(resized_key)
    if resized is None:
        resized = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)
        FISH_RESIZED_SPRITE_CACHE[resized_key] = resized

    h, w = resized.shape[:2]
    grid = FISH_WARP_GRID_CACHE.get((w, h))
    if grid is None:
        base_x, base_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        y_norm = base_y / max(1.0, float(h - 1))
        body_weight = np.clip((y_norm - 0.24) / 0.42, 0.0, 1.0)
        body_weight = body_weight * body_weight * (3.0 - 2.0 * body_weight)
        tail_weight = np.clip((y_norm - 0.62) / 0.38, 0.0, 1.0)
        tail_weight = tail_weight * tail_weight * (3.0 - 2.0 * tail_weight)
        tail_weight = (body_weight * 0.32 + tail_weight * 0.68).astype(np.float32)
        grid = (base_x, base_y, y_norm, tail_weight)
        FISH_WARP_GRID_CACHE[(w, h)] = grid
    base_x, base_y, y_norm, tail_weight = grid
    move_speed = math.hypot(
        float(_fish_value(fish, "vx", 0.0)),
        float(_fish_value(fish, "vy", 0.0)),
    )
    speed_bonus = min(FISH_WIGGLE_SPEED_MAX_BONUS, (move_speed / max(1.0, float(radius))) * FISH_WIGGLE_SPEED_BY_MOVE)
    phase = float(now) * (FISH_WIGGLE_SPEED + speed_bonus) + float(_fish_value(fish, "wiggle_phase", _fish_value(fish, "phase", 0.0)))
    amplitude = max(1.5, radius * 0.18)

    wave = np.sin(phase - y_norm * math.tau * FISH_WIGGLE_BODY_WAVE) * amplitude * tail_weight
    map_x = base_x - wave.astype(np.float32)

    return cv2.remap(
        resized,
        map_x,
        base_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _rotate_rgba(image, angle_deg):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def add_gold_sparkles(rgba, phase_bucket):
    if rgba is None or rgba.size == 0 or rgba.shape[2] < 4:
        return rgba

    image = rgba.copy()
    h, w = image.shape[:2]
    sparkle_points = (
        (0.32, 0.25, 0),
        (0.66, 0.34, 2),
        (0.48, 0.56, 4),
    )
    for x_ratio, y_ratio, offset in sparkle_points:
        pulse = (phase_bucket + offset) % 8
        if pulse >= 5:
            continue

        alpha = int(165 + pulse * 16)
        radius = max(2, int(min(w, h) * (0.026 + pulse * 0.003)))
        cx = int(w * x_ratio)
        cy = int(h * y_ratio)
        color = (255, 248, 220, alpha)
        cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)
        arm = radius * 3
        cv2.line(image, (cx - arm, cy), (cx + arm, cy), color, max(1, radius // 2), cv2.LINE_AA)
        cv2.line(image, (cx, cy - arm), (cx, cy + arm), color, max(1, radius // 2), cv2.LINE_AA)

    return image


def make_gold_outline(rgba):
    if rgba is None or rgba.size == 0 or rgba.shape[2] < 4:
        return None

    alpha = rgba[:, :, 3]
    k = max(3, int(GOLD_FISH_OUTLINE_SIZE))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    outline_alpha = cv2.dilate(alpha, kernel, iterations=1)
    outline_alpha = cv2.subtract(outline_alpha, alpha)
    if np.max(outline_alpha) <= 0:
        return None

    outline = np.zeros_like(rgba)
    outline[:, :, :3] = (35, 220, 255)
    outline[:, :, 3] = np.minimum(outline_alpha, GOLD_FISH_OUTLINE_ALPHA).astype(np.uint8)
    return outline


def alpha_blend(frame, rgba, center_x, center_y):
    if rgba is None or rgba.size == 0 or rgba.shape[2] < 4:
        return False

    h, w = rgba.shape[:2]
    x1 = int(round(center_x - w / 2))
    y1 = int(round(center_y - h / 2))
    x2 = x1 + w
    y2 = y1 + h

    frame_h, frame_w = frame.shape[:2]
    roi_x1 = max(0, x1)
    roi_y1 = max(0, y1)
    roi_x2 = min(frame_w, x2)
    roi_y2 = min(frame_h, y2)
    if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
        return False

    sprite_x1 = roi_x1 - x1
    sprite_y1 = roi_y1 - y1
    sprite_x2 = sprite_x1 + (roi_x2 - roi_x1)
    sprite_y2 = sprite_y1 + (roi_y2 - roi_y1)

    sprite_roi = rgba[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
    alpha = sprite_roi[:, :, 3:4].astype(np.uint16)
    if np.max(alpha) <= 0:
        return False

    frame_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.uint16)
    sprite_rgb = sprite_roi[:, :, :3].astype(np.uint16)
    blended = (sprite_rgb * alpha + frame_roi * (255 - alpha) + 127) // 255
    frame[roi_y1:roi_y2, roi_x1:roi_x2] = blended.astype(np.uint8)
    return True


def load_water_grass_sprites():
    global WATER_GRASS_SPRITES
    if WATER_GRASS_SPRITES is not None:
        return WATER_GRASS_SPRITES

    img_dir = Path(__file__).resolve().parent.parent / "img"
    sprites = {}
    for key, filename in (("grass1", "water_grass_1.png"), ("grass2", "water_grass_2.png")):
        sprite = cv2.imread(str(img_dir / filename), cv2.IMREAD_UNCHANGED)
        if sprite is None:
            continue
        if sprite.ndim == 2:
            sprite = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGRA)
        elif sprite.shape[2] == 3:
            alpha = np.full(sprite.shape[:2] + (1,), 255, dtype=np.uint8)
            sprite = np.concatenate([sprite, alpha], axis=2)
        sprites[key] = sprite

    WATER_GRASS_SPRITES = sprites
    return WATER_GRASS_SPRITES


def draw_water_grass(screen):
    sprites = WATER_GRASS_SPRITES if WATER_GRASS_SPRITES is not None else load_water_grass_sprites()
    if not sprites:
        return

    h, w = screen.shape[:2]
    placements = (
        ("grass1", 0.57, 0.62, 0.33),
        ("grass2", 0.09, 0.17, 0.34),
    )
    now = time.perf_counter()
    for index, (key, x_ratio, y_ratio, height_ratio) in enumerate(placements):
        sprite = sprites.get(key)
        if sprite is None:
            continue

        target_h = max(24, int(h * height_ratio))
        src_h, src_w = sprite.shape[:2]
        target_w = max(16, int(target_h * src_w / max(1, src_h)))
        cache_key = (key, target_w, target_h)
        resized = WATER_GRASS_RESIZED_CACHE.get(cache_key)
        if resized is None:
            resized = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized[:, :, :3] = cv2.convertScaleAbs(resized[:, :, :3], alpha=WATER_GRASS_DARKEN_ALPHA, beta=0)
            WATER_GRASS_RESIZED_CACHE[cache_key] = resized

        phase = now * WATER_GRASS_SWAY_FPS + index * 1.7
        sway_x = math.sin(phase) * w * WATER_GRASS_SWAY_X_RATIO
        sway_y = math.sin(phase * 0.7 + 0.8) * h * WATER_GRASS_SWAY_Y_RATIO
        alpha_blend(screen, resized, w * x_ratio + sway_x, h * y_ratio + sway_y)


def draw_fish_shadow(frame, center_x, center_y, sprite_w, sprite_h, radius):
    h, w = frame.shape[:2]
    shadow_w = max(10, int(sprite_w * 0.42))
    shadow_h = max(4, int(sprite_h * 0.13))
    shadow_x = int(round(center_x))
    shadow_y = int(round(center_y + radius * 0.58))

    x1 = max(0, shadow_x - shadow_w)
    y1 = max(0, shadow_y - shadow_h)
    x2 = min(w, shadow_x + shadow_w + 1)
    y2 = min(h, shadow_y + shadow_h + 1)
    if x1 >= x2 or y1 >= y2:
        return

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (shadow_x - x1, shadow_y - y1),
        (shadow_w, shadow_h),
        0,
        0,
        360,
        70,
        -1,
        cv2.LINE_AA,
    )
    alpha = mask[:, :, None].astype(np.uint16)
    roi = frame[y1:y2, x1:x2].astype(np.uint16)
    shadow_color = np.array([18, 26, 30], dtype=np.uint16)
    frame[y1:y2, x1:x2] = ((shadow_color * alpha + roi * (255 - alpha) + 127) // 255).astype(np.uint8)


def draw_fish_sprite(frame, fish, now):
    sprites = FISH_SPRITES if FISH_SPRITES is not None else load_fish_sprites()
    key = _fish_sprite_key(fish)
    sprite = sprites.get(key)
    if sprite is None and key == "bad":
        sprite = sprites.get("bad_mark")
    if sprite is None:
        return False

    x = _fish_value(fish, "x")
    y = _fish_value(fish, "y")
    if x is None or y is None:
        pos = _fish_value(fish, "pos") or _fish_value(fish, "position")
        if pos is not None and len(pos) >= 2:
            x, y = pos[0], pos[1]
    if x is None or y is None:
        return False

    vx = float(_fish_value(fish, "vx", 0.0))
    vy = float(_fish_value(fish, "vy", 0.0))
    if abs(vx) + abs(vy) < 0.001:
        angle = float(_fish_value(fish, "angle", 0.0))
    else:
        angle = -(math.degrees(math.atan2(vy, vx)) + 90.0)

    radius = max(8, int(_fish_value(fish, "radius", 24)))
    angle_bucket = int(round(angle / FISH_ANGLE_BUCKET_DEG))
    frame_bucket = int(float(now) * FISH_ANIMATION_FPS)
    cache_key = (id(fish), key, radius, angle_bucket, frame_bucket)
    rotated = FISH_RENDER_CACHE.get(cache_key)
    if rotated is None:
        warped = warp_fish_sprite(sprite, fish, frame_bucket / FISH_ANIMATION_FPS)
        rotated = _rotate_rgba(warped, angle_bucket * FISH_ANGLE_BUCKET_DEG)
        if key == "rare":
            rotated = add_gold_sparkles(rotated, frame_bucket)
        prune_fish_render_cache(frame_bucket)
        FISH_RENDER_CACHE[cache_key] = rotated

    shadow_key = _fish_shadow_key(key)
    shadow_sprite = sprites.get(shadow_key)
    if FISH_SHADOW_ENABLED and shadow_sprite is not None:
        shadow_size_multiplier = GOLD_FISH_SHADOW_SCALE if key == "rare" else 1.0
        shadow_frame_bucket = frame_bucket // max(1, int(FISH_SHADOW_ANIMATION_DIVISOR))
        shadow_cache_key = (id(fish), shadow_key, radius, angle_bucket, shadow_frame_bucket, shadow_size_multiplier)
        shadow_rotated = FISH_RENDER_CACHE.get(shadow_cache_key)
        if shadow_rotated is None:
            shadow_time = (shadow_frame_bucket * FISH_SHADOW_ANIMATION_DIVISOR) / FISH_ANIMATION_FPS
            shadow_warped = warp_fish_sprite(
                shadow_sprite,
                fish,
                shadow_time,
                shadow_key,
                shadow_size_multiplier,
            )
            shadow_rotated = _rotate_rgba(shadow_warped, angle_bucket * FISH_ANGLE_BUCKET_DEG)
            prune_fish_render_cache(frame_bucket)
            FISH_RENDER_CACHE[shadow_cache_key] = shadow_rotated
        shadow_offset = max(3, int(radius * 0.34))
        alpha_blend(frame, shadow_rotated, float(x) + shadow_offset, float(y) + shadow_offset)
    return alpha_blend(frame, rotated, float(x), float(y))


def draw_fish(frame, fish, *args, **kwargs):
    now = kwargs.get("now", kwargs.get("t", None))
    if now is None:
        for value in args:
            if isinstance(value, (int, float)):
                now = value
                break
    if now is None:
        now = time.time()

    try:
        if draw_fish_sprite(frame, fish, now):
            return None
    except Exception:
        pass
    return _draw_fish_shape_fallback(frame, fish, *args, **kwargs)


def rebalance_fish_spawn_rates():
    if "FISH_TYPES" not in globals():
        return

    spawn_rates = {
        "normal": 0.47,
        "red": 0.33,
        "rare": 0.20,
    }
    rate_keys = ("weight", "spawn_weight", "probability", "prob", "chance")

    for fish_type in FISH_TYPES:
        if not isinstance(fish_type, dict):
            continue
        name = str(fish_type.get("name", "")).strip().lower()
        if "bad" in name or "black" in name:
            continue
        if "rare" in name or "special" in name:
            rate = spawn_rates["rare"]
        elif "red" in name:
            rate = spawn_rates["red"]
        else:
            rate = spawn_rates["normal"]

        has_rate_key = False
        for key in rate_keys:
            if key in fish_type:
                fish_type[key] = rate
                has_rate_key = True
        if not has_rate_key:
            fish_type["weight"] = rate


rebalance_fish_spawn_rates()


if __name__ == "__main__":
    main()

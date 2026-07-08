import argparse
import csv
import random
import time
from collections import Counter, deque
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
    parser.add_argument("--hit-effect-sec", type=float, default=0.30)
    parser.add_argument("--target-life-sec", type=float, default=5.0)
    parser.add_argument("--target-overlap-sec", type=float, default=3.0)
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
    draw_grid(screen)
    cx, cy = w // 2, h // 2
    r = scaled(radius, 1080, h)

    cv2.circle(screen, (cx, cy), r + scaled(14, 1080, h), (255, 255, 255), scaled(4, 1080, h), cv2.LINE_AA)
    cv2.circle(screen, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)

    draw_centered_text(screen, "Look at the center dot", scaled(120, 1080, h), scale=1.5 * h / 1080, thickness=scaled(4, 1080, h))
    draw_centered_text(screen, "Press Enter to start", scaled(185, 1080, h), color=(0, 255, 255), scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))
    draw_centered_text(screen, "Press q to quit", scaled(235, 1080, h), scale=0.85 * h / 1080, thickness=scaled(2, 1080, h))
    return screen


def draw_calibration_screen(w, h, radius, remain_sec, elapsed, ignore_sec):
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    draw_grid(screen)
    cx, cy = w // 2, h // 2
    r = scaled(radius, 1080, h)

    cv2.circle(screen, (cx, cy), r + scaled(14, 1080, h), (255, 255, 255), scaled(4, 1080, h), cv2.LINE_AA)
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


def choose_next_target(exclude_areas=None):
    """除外領域を避けて、次の的の領域を選ぶ。

    exclude_areas で全領域が埋まっている場合だけ、重なりを許して選ぶ。
    通常運用では最大2体までなので、ここで重なることはない。
    """
    all_areas = [0, 1, 2, 3]
    exclude_areas = set(exclude_areas or [])
    candidates = [area for area in all_areas if area not in exclude_areas]
    if not candidates:
        candidates = all_areas
    return random.choice(candidates)


def spawn_target(exclude_areas, args):
    area = choose_next_target(exclude_areas)
    bonus_chance = min(1.0, max(0.0, float(args.bonus_chance)))
    return {
        "area": area,
        "is_bonus": random.random() < bonus_chance,
        "spawn_time": time.perf_counter(),
    }


def update_targets(targets, args):
    """的を更新する。

    仕様:
    - 的は最大2体まで。
    - 各的は5秒で消える。
    - 的が1体だけで、その的が3秒以上残っていれば、別位置に2体目を出す。
    - 表示中の的と同じ領域には新しい的を出さない。
    """
    now = time.perf_counter()
    life_sec = max(0.1, float(args.target_life_sec))
    overlap_sec = max(0.0, min(float(args.target_overlap_sec), life_sec))
    max_targets = 2

    # 先に5秒経過した的を消す。
    targets[:] = [t for t in targets if now - t["spawn_time"] < life_sec]

    # 念のため、同じ領域に複数の的がある場合は新しい方を残して整理する。
    unique_by_area = {}
    for target in targets:
        area = target["area"]
        if area not in unique_by_area or target["spawn_time"] > unique_by_area[area]["spawn_time"]:
            unique_by_area[area] = target
    targets[:] = sorted(unique_by_area.values(), key=lambda t: t["spawn_time"])[:max_targets]

    # すべて消えた場合は1体出す。
    if not targets:
        targets.append(spawn_target([], args))
        return targets

    # 的が1体だけ、かつ3秒以上経過していれば、別位置に2体目を出す。
    if len(targets) < max_targets:
        oldest = min(targets, key=lambda t: t["spawn_time"])
        if now - oldest["spawn_time"] >= overlap_sec:
            active_areas = [t["area"] for t in targets]
            targets.append(spawn_target(active_areas, args))

    # 念のため最大2体に制限する。
    targets[:] = sorted(targets, key=lambda t: t["spawn_time"])[:max_targets]
    return targets


def draw_enemy_mark(screen, area_id, args, is_bonus=False):
    h, w = screen.shape[:2]
    cx, cy = get_area_center(w, h, area_id)
    color = (0, 255, 255) if is_bonus else get_enemy_color(args.enemy_color)
    r = scaled(args.enemy_radius, 1080, h)
    if is_bonus:
        r = int(r * 1.15)
    cross = int(r * 0.65)
    th = scaled(7, 1080, h)

    cv2.circle(screen, (cx, cy), r, color, th, cv2.LINE_AA)
    cv2.line(screen, (cx - cross, cy), (cx + cross, cy), color, th, cv2.LINE_AA)
    cv2.line(screen, (cx, cy - cross), (cx, cy + cross), color, th, cv2.LINE_AA)

    if is_bonus:
        cv2.circle(screen, (cx, cy), int(r * 1.28), color, scaled(3, 1080, h), cv2.LINE_AA)
        draw_centered_text(screen, "+3", cy + r + scaled(58, 1080, h), color=color, scale=0.9 * h / 1080, thickness=scaled(3, 1080, h))


def draw_gaze_area_frame(screen, area_id, blink=False):
    if area_id is None:
        return
    h, w = screen.shape[:2]
    x1, y1, x2, y2 = get_area_rect(w, h, area_id)
    margin = scaled(10, 1080, h)
    th = scaled(8, 1080, h)

    # 通常時は白枠、瞬き検出中だけマゼンタ枠にする。
    # BGR: white=(255,255,255), magenta=(255,0,255)
    color = (255, 0, 255) if blink else (255, 255, 255)

    cv2.rectangle(
        screen,
        (x1 + margin, y1 + margin),
        (x2 - margin, y2 - margin),
        color,
        th,
        cv2.LINE_AA,
    )


def draw_game_screen(w, h, targets, gaze_area, score, remain_sec, args,
                     blink=False, head_warning=False, hit_area=None, hit_visible=False):
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    draw_grid(screen)

    draw_gaze_area_frame(screen, gaze_area, blink=blink)

    for target in targets:
        draw_enemy_mark(screen, target["area"], args, is_bonus=target["is_bonus"])

    put_text(screen, f"SCORE: {score}", scaled(55, 1080, h), scale=1.2 * h / 1080, thickness=scaled(3, 1080, h), x=scaled(35, 1920, w))
    time_text = f"TIME: {remain_sec:.1f}"
    (tw, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * h / 1080, scaled(3, 1080, h))
    cv2.putText(screen, time_text, (w - tw - scaled(35, 1920, w), scaled(55, 1080, h)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * h / 1080, (255, 255, 255), scaled(3, 1080, h), cv2.LINE_AA)

    draw_centered_text(screen, f"TARGET: {len(targets)}", scaled(55, 1080, h), color=(0, 255, 255), scale=0.9 * h / 1080, thickness=scaled(2, 1080, h))

    if hit_visible and hit_area is not None:
        cx, cy = get_area_center(w, h, hit_area)
        draw_centered_text(screen, "HIT!", cy - scaled(120, 1080, h), color=(0, 255, 255), scale=1.8 * h / 1080, thickness=scaled(5, 1080, h))

    if blink:
        draw_centered_text(screen, "SHOT", h - scaled(90, 1080, h), color=(0, 255, 255), scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))
    elif head_warning:
        draw_centered_text(screen, "Please keep your head still", h - scaled(90, 1080, h), color=(0, 0, 255), scale=1.0 * h / 1080, thickness=scaled(3, 1080, h))

    put_text(screen, "Look at the mark and blink to shoot / Press q to quit", h - scaled(35, 1080, h), scale=0.8 * h / 1080, thickness=scaled(2, 1080, h))
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
    draw_grid(screen)
    draw_centered_text(screen, "TIME UP!", scaled(210, 1080, h), color=(0, 255, 255), scale=2.1 * h / 1080, thickness=scaled(6, 1080, h))
    draw_centered_text(screen, f"SCORE: {score}", scaled(325, 1080, h), scale=1.7 * h / 1080, thickness=scaled(5, 1080, h))

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

                key = cv2.waitKey(1) & 0xFF
                if key in (13, 10) and baseline_info is not None:
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            center_feature, center_info, quit_requested = run_center_calibration(
                cap, face_mesh, args, calib_smoother
            )
            if quit_requested:
                break
            if center_feature is None:
                continue
            if center_info is None:
                center_info = baseline_info

            base_eye_x = center_info["eye_center_x"]
            base_eye_y = center_info["eye_center_y"]
            base_eye_dist = center_info["inter_eye_dist"]

            realtime_smoother.reset()
            pred_history = deque(maxlen=max(1, args.majority_window))
            last_valid_area = None
            post_blink_count = 0
            prev_blink = False
            score = 0
            targets = [spawn_target([], args)]
            hit_area = None
            hit_effect_until = 0.0
            game_start_time = time.perf_counter()
            
            fps_start = time.perf_counter()
            fps_frames = 0
            fps = 0.0

            while True:
                game_elapsed = time.perf_counter() - game_start_time
                game_remain = max(0.0, args.game_sec - game_elapsed)
                if game_elapsed >= args.game_sec:
                    break

                targets = update_targets(targets, args)

                frame, feature, info = process_frame(cap, face_mesh, args, realtime_smoother, args.realtime_beta)

                landmark_view = draw_landmark_view(frame, info)
                if landmark_view is not None:
                    cv2.imshow("camera_landmarks", landmark_view)
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

                    if blink_trigger and last_valid_area is not None:
                        hit_target = None
                        for target in targets:
                            if target["area"] == last_valid_area:
                                hit_target = target
                                break
                        if hit_target is not None:
                            score += int(args.bonus_score if hit_target["is_bonus"] else args.normal_score)
                            hit_area = hit_target["area"]
                            hit_effect_until = time.perf_counter() + args.hit_effect_sec
                            targets.remove(hit_target)
                            if len(targets) < 2:
                                targets.append(spawn_target([t["area"] for t in targets] + [hit_area], args))

                    if blink_now:
                        post_blink_count = args.post_blink_hold_frames
                    elif post_blink_count > 0:
                        post_blink_count -= 1
                    else:
                        raw_area = classify_pure4(dx, dy, args.dead_x, args.dead_y)
                        if raw_area is not None:
                            pred_history.append(raw_area)
                            last_valid_area = Counter(pred_history).most_common(1)[0][0]

                    prev_blink = blink_now
                else:
                    prev_blink = False

                screen = draw_game_screen(
                    args.screen_width,
                    args.screen_height,
                    targets,
                    last_valid_area,
                    score,
                    game_remain,
                    args,
                    blink=blink_now,
                    head_warning=head_warning,
                    hit_area=hit_area,
                    hit_visible=time.perf_counter() < hit_effect_until,
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


if __name__ == "__main__":
    main()

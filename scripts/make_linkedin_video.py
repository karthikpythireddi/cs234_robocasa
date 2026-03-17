#!/usr/bin/env python3
"""
make_linkedin_video.py

Creates a high-quality LinkedIn demo video from GR00T eval rollouts.
For each task, plays Base / DPO / RWR side-by-side simultaneously,
picking the best available episode (SUCCESS preferred, else first).

Usage (run on Lightning AI after eval):
    python scripts/make_linkedin_video.py \
        --eval_dir outputs/eval \
        --output_dir outputs/linkedin_videos \
        --model_order base dpo rwr

    # Single task:
    python scripts/make_linkedin_video.py \
        --eval_dir outputs/eval \
        --task PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        --output_dir outputs/linkedin_videos
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from pathlib import Path


# ── Display config ────────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "base": "Base  GR00T N1.6",
    "dpo":  "DPO",
    "rwr":  "RLHF / RWR",
    "ppo":  "PPO",
}
MODEL_COLORS_BGR = {
    "base": (176, 114,  76),   # steel blue
    "dpo":  ( 82, 132, 221),   # orange
    "rwr":  (104, 168,  85),   # green
    "ppo":  ( 82,  78, 196),   # red
}
TASK_SHORT = {
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env":
        "Bottle → Cabinet",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env":
        "Cutting Board → Basket",
    "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env":
        "Plate → Bowl",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env":
        "Tray → Pot",
}

PANEL_W    = 480    # width per model panel
HEADER_H   = 52     # per-model label bar height
TITLE_H    = 60     # top banner height
FOOTER_H   = 36     # bottom strip height
FPS_OUT    = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_video(video_dir: str, episode: int = 0) -> tuple:
    """Return (path, is_success).
    Picks the specific episode number so all models see the exact same environment
    (seed = episode * 7 + 42 is hardcoded in eval_policy.py, same for every model).
    Falls back to ep00 or first available if the requested episode is missing.
    """
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        return None, False

    ep_tag = f"ep{episode:02d}_"
    ep_videos = [v for v in videos if os.path.basename(v).startswith(ep_tag)]
    if not ep_videos:
        ep_videos = [v for v in videos if os.path.basename(v).startswith("ep00_")]
    if not ep_videos:
        ep_videos = videos

    v = ep_videos[0]
    return v, "SUCCESS" in os.path.basename(v)


def load_success_rate(eval_dir: str, model: str, task: str) -> int:
    fp = os.path.join(eval_dir, model, task, "eval_results.json")
    if os.path.isfile(fp):
        with open(fp) as f:
            d = json.load(f)
        sr = d.get("summary", {}).get("success_rate", 0)
        return int(round(sr * 100))
    return -1


def read_video_frames(path: str, target_w: int, target_h: int):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()
    return frames


def make_header(w: int, model: str, sr: int, is_success: bool) -> np.ndarray:
    bar = np.zeros((HEADER_H, w, 3), dtype=np.uint8)
    bar[:] = MODEL_COLORS_BGR[model]
    label = MODEL_DISPLAY.get(model, model)
    sr_str = f"{sr}%" if sr >= 0 else "N/A"
    icon   = "✓" if is_success else "✗"
    text   = f"{label}   {sr_str} avg success   {icon}"

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = (HEADER_H + th) // 2 - 2
    cv2.putText(bar, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return bar


def make_title(total_w: int, task_label: str) -> np.ndarray:
    bar = np.zeros((TITLE_H, total_w, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    # Main title
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    title = f"GR00T Humanoid  |  {task_label}"
    (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
    x = (total_w - tw) // 2
    cv2.putText(bar, title, (x, th + 6), font, scale, (240, 240, 240), thick, cv2.LINE_AA)

    # Subtitle
    sub = "Preference Optimization for Continuous Robotic Control  |  Stanford CS234"
    sfont, sscale, sthick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (stw, _), _ = cv2.getTextSize(sub, sfont, sscale, sthick)
    sx = (total_w - stw) // 2
    cv2.putText(bar, sub, (sx, th + 30), sfont, sscale, (160, 160, 160), sthick, cv2.LINE_AA)
    return bar


def make_footer(total_w: int) -> np.ndarray:
    bar = np.zeros((FOOTER_H, total_w, 3), dtype=np.uint8)
    bar[:] = (18, 18, 18)
    text = "Karthik Pythireddi   |   Stanford CS234: Reinforcement Learning"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (total_w - tw) // 2
    y = (FOOTER_H + th) // 2
    cv2.putText(bar, text, (x, y), font, scale, (140, 140, 140), thick, cv2.LINE_AA)
    return bar


# ── Transition / intro cards ──────────────────────────────────────────────────

def make_intro_card(total_w: int, total_h: int) -> np.ndarray:
    card = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    card[:] = (18, 18, 18)

    title = "GR00T N1.6 Humanoid Robot"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
    (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
    cv2.putText(card, title, ((total_w - tw) // 2, total_h // 2 - 20),
                font, scale, (240, 240, 240), thick, cv2.LINE_AA)

    sub = "Preference Optimization for Continuous Robotic Control  |  Stanford CS234"
    sfont, sscale, sthick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (stw, _), _ = cv2.getTextSize(sub, sfont, sscale, sthick)
    cv2.putText(card, sub, ((total_w - stw) // 2, total_h // 2 + th),
                sfont, sscale, (120, 120, 120), sthick, cv2.LINE_AA)
    return card


def make_transition_card(total_w: int, total_h: int, task_label: str, task_num: int, n_tasks: int) -> np.ndarray:
    card = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    card[:] = (18, 18, 18)

    pill = f"Task {task_num} of {n_tasks}"
    pfont, pscale, pthick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (pw, ph), _ = cv2.getTextSize(pill, pfont, pscale, pthick)
    cv2.putText(card, pill, ((total_w - pw) // 2, total_h // 2 - 40),
                pfont, pscale, (100, 100, 100), pthick, cv2.LINE_AA)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2
    (tw, th), _ = cv2.getTextSize(task_label, font, scale, thick)
    cv2.putText(card, task_label, ((total_w - tw) // 2, total_h // 2 + th // 2),
                font, scale, (240, 240, 240), thick, cv2.LINE_AA)

    sub = "Base GR00T  vs  DPO  vs  RLHF/RWR  vs  PPO"
    sfont, sscale, sthick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (stw, _), _ = cv2.getTextSize(sub, sfont, sscale, sthick)
    cv2.putText(card, sub, ((total_w - stw) // 2, total_h // 2 + th // 2 + 36),
                sfont, sscale, (120, 120, 120), sthick, cv2.LINE_AA)
    return card


# ── Main per-task video builder ───────────────────────────────────────────────

def collect_task_frames(eval_dir: str, task: str, models: list, episode: int = 0) -> tuple:
    """Load and composite all frames for a task. Returns (frames_list, (total_w, total_h)) or (None, None)."""
    task_label = TASK_SHORT.get(task, task[:40])
    model_data = {}
    frame_h = None

    for model in models:
        video_dir = os.path.join(eval_dir, model, task, "videos")
        vpath, is_success = pick_video(video_dir, episode=episode)
        sr = load_success_rate(eval_dir, model, task)
        if vpath is None:
            model_data[model] = {"frames": [], "sr": sr, "success": False}
            continue
        if frame_h is None:
            cap = cv2.VideoCapture(vpath)
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            frame_h = int(native_h * PANEL_W / native_w)
        frames = read_video_frames(vpath, PANEL_W, frame_h)
        model_data[model] = {"frames": frames, "sr": sr, "success": is_success}

    if frame_h is None:
        return None, None

    total_w   = PANEL_W * len(models)
    total_h   = TITLE_H + HEADER_H + frame_h + FOOTER_H
    title_bar  = make_title(total_w, task_label)
    footer_bar = make_footer(total_w)
    blank      = np.zeros((frame_h, PANEL_W, 3), dtype=np.uint8)
    n_frames   = max(len(d["frames"]) for d in model_data.values()) or 1

    composed = []
    for fi in range(n_frames):
        panels = []
        for model in models:
            d = model_data[model]
            frames = d["frames"]
            frame  = frames[min(fi, len(frames) - 1)] if frames else blank.copy()
            header = make_header(PANEL_W, model, d["sr"], d["success"])
            panels.append(np.vstack([header, frame]))
        composed.append(np.vstack([title_bar, np.hstack(panels), footer_bar]))

    return composed, (total_w, total_h)


def make_task_video(eval_dir: str, task: str, models: list, output_path: str, episode: int = 0):
    print(f"\n[{task}]")
    task_label = TASK_SHORT.get(task, task[:40])

    # Load frames for each model
    model_data = {}
    frame_h = None
    for model in models:
        video_dir = os.path.join(eval_dir, model, task, "videos")
        vpath, is_success = pick_video(video_dir, episode=episode)
        sr = load_success_rate(eval_dir, model, task)
        if vpath is None:
            print(f"  [{model}] no video found — using blank")
            model_data[model] = {"frames": [], "sr": sr, "success": False}
            continue
        print(f"  [{model}] {os.path.basename(vpath)}  sr={sr}%  success={is_success}")

        # Probe native size from first model that has a video
        if frame_h is None:
            cap = cv2.VideoCapture(vpath)
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            frame_h = int(native_h * PANEL_W / native_w)

        frames = read_video_frames(vpath, PANEL_W, frame_h)
        model_data[model] = {"frames": frames, "sr": sr, "success": is_success}

    if frame_h is None:
        print("  No videos found for any model — skipping task")
        return

    n_frames  = max(len(d["frames"]) for d in model_data.values()) or 1
    total_w   = PANEL_W * len(models)
    total_h   = TITLE_H + HEADER_H + frame_h + FOOTER_H

    title_bar  = make_title(total_w, task_label)
    footer_bar = make_footer(total_w)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS_OUT, (total_w, total_h))

    blank = np.zeros((frame_h, PANEL_W, 3), dtype=np.uint8)

    for fi in range(n_frames):
        panels = []
        for model in models:
            d = model_data[model]
            frames = d["frames"]
            frame  = frames[min(fi, len(frames) - 1)] if frames else blank
            header = make_header(PANEL_W, model, d["sr"], d["success"])
            panels.append(np.vstack([header, frame]))

        row  = np.hstack(panels)
        full = np.vstack([title_bar, row, footer_bar])
        writer.write(full)

    writer.release()
    print(f"  Saved → {output_path}")


# ── Combined video ────────────────────────────────────────────────────────────

def make_combined_video(eval_dir: str, models: list, output_path: str,
                        transition_secs: float = 2.0, intro_secs: float = 2.5):
    tasks = list(TASK_SHORT.keys())

    print("Loading all task frames...")
    all_task_frames = {}
    dims = None
    for task in tasks:
        frames, d = collect_task_frames(eval_dir, task, models)
        if frames:
            all_task_frames[task] = frames
            dims = d
            print(f"  {TASK_SHORT[task]}: {len(frames)} frames")

    if dims is None:
        print("[error] No valid task frames found")
        return

    total_w, total_h = dims
    intro_n  = int(intro_secs * FPS_OUT)
    trans_n  = int(transition_secs * FPS_OUT)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS_OUT, (total_w, total_h))

    intro = make_intro_card(total_w, total_h)
    print(f"\nWriting intro ({intro_n} frames)...")
    for _ in range(intro_n):
        writer.write(intro)

    for i, task in enumerate(tasks):
        card = make_transition_card(total_w, total_h, TASK_SHORT[task], i + 1, len(tasks))
        print(f"Writing transition: '{TASK_SHORT[task]}' ({trans_n} frames)...")
        for _ in range(trans_n):
            writer.write(card)

        frames = all_task_frames.get(task, [])
        print(f"Writing task frames: {len(frames)}...")
        for f in frames:
            writer.write(f)

    writer.release()
    print(f"\nCombined video → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",    default="outputs/eval")
    parser.add_argument("--output_dir",  default="outputs/linkedin_videos")
    parser.add_argument("--model_order", nargs="+", default=["base", "dpo", "rwr", "ppo"])
    parser.add_argument("--task",        default=None,
                        help="Single task name. If omitted, all 4 tasks are generated.")
    parser.add_argument("--combined",    action="store_true",
                        help="Generate single combined video of all tasks")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.combined:
        out = os.path.join(args.output_dir, "groot_all_tasks_combined.mp4")
        make_combined_video(args.eval_dir, args.model_order, out)
    elif args.task:
        tasks = [args.task]
        for task in tasks:
            safe = task.replace("_GR1ArmsAndWaistFourierHands_Env", "")
            out  = os.path.join(args.output_dir, f"groot_{safe}.mp4")
            make_task_video(args.eval_dir, task, args.model_order, out)
    else:
        tasks = list(TASK_SHORT.keys())
        for task in tasks:
            safe = task.replace("_GR1ArmsAndWaistFourierHands_Env", "")
            out  = os.path.join(args.output_dir, f"groot_{safe}.mp4")
            make_task_video(args.eval_dir, task, args.model_order, out)

    print(f"\nAll done. Videos in: {args.output_dir}")


if __name__ == "__main__":
    main()

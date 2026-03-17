#!/usr/bin/env python3
"""
make_eval_montage.py

Create side-by-side comparison figures from evaluation videos.
For each task, extracts key frames from base/DPO/RWR/PPO rollouts
and arranges them into a single montage figure for the report.

Style: LIBERO-style colored header bars per model row with success rate
and checkmark/X indicator, green=success, red=fail.

Usage:
    python scripts/make_eval_montage.py \
        --eval_dir outputs/eval \
        --output_dir outputs/figures \
        --model_order base dpo rwr ppo \
        --n_frames 5
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from pathlib import Path


MODEL_DISPLAY = {
    "base": "Base (GR00T N1.6)",
    "dpo": "DPO",
    "rwr": "RLHF",
    "ppo": "PPO",
}

# Colors in RGB
MODEL_COLORS = {
    "base": (76, 114, 176),    # steel blue
    "dpo": (221, 132, 82),     # orange
    "rwr": (85, 168, 104),     # green
    "ppo": (196, 78, 82),      # red
}

SUCCESS_COLOR_BG = (46, 139, 87)    # sea green
FAIL_COLOR_BG = (178, 34, 34)       # firebrick red
HEADER_HEIGHT = 32


def extract_frames_from_video(video_path: str, n_frames: int = 5) -> list:
    """Extract n_frames evenly spaced frames from a video.

    GR00T videos show dual ego views (padded + cropped) side by side,
    which is the actual observation space of the humanoid model.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Warning: cannot open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def pick_best_video(video_dir: str, prefer_success: bool = True) -> tuple:
    """Pick a representative video from the directory.
    Prefers successful episodes if available.
    Returns (video_path, is_success) or (None, False)."""
    if not os.path.isdir(video_dir):
        return None, False

    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        return None, False

    if prefer_success:
        success_renamed = [v for v in videos if "SUCCESS" in os.path.basename(v)]
        if success_renamed:
            return success_renamed[0], True
        success_s1 = [v for v in videos if "_s1" in os.path.basename(v)]
        if success_s1:
            return success_s1[0], True

    # Fall back to first video (likely a failure)
    basename = os.path.basename(videos[0])
    is_success = "SUCCESS" in basename or "_s1" in basename
    return videos[0], is_success


def load_success_rate(eval_dir: str, model: str, task: str) -> float | None:
    """Load success rate for a specific model/task from eval_results.json."""
    fp = os.path.join(eval_dir, model, task, "eval_results.json")
    if os.path.isfile(fp):
        with open(fp) as f:
            data = json.load(f)
        return data["summary"]["success_rate"]
    return None


def make_header_bar(width: int, model: str, success_rate: float | None,
                    is_success: bool) -> np.ndarray:
    """Create a LIBERO-style colored header bar for a model row.

    Format: "MODEL_NAME  XX.X%  checkmark/X"
    Background color: green if episode shown is success, red if fail.
    """
    bar = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)

    # Background color based on whether the shown episode succeeded
    bg_color = SUCCESS_COLOR_BG if is_success else FAIL_COLOR_BG
    bar[:] = bg_color

    font = cv2.FONT_HERSHEY_SIMPLEX
    display_name = MODEL_DISPLAY.get(model, model.upper())

    # Build label text
    if success_rate is not None:
        pct = f"{success_rate * 100:.0f}%"
        indicator = "+" if is_success else "x"
        label = f"{display_name}  {pct}  {indicator}"
    else:
        indicator = "+" if is_success else "x"
        label = f"{display_name}  {indicator}"

    # Draw text centered-left with good visibility
    font_scale = 0.65
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x = 10
    y = (HEADER_HEIGHT + text_size[1]) // 2

    # White text on colored background
    cv2.putText(bar, label, (x, y), font, font_scale, (255, 255, 255), thickness)

    return bar


def make_montage_for_task(
    task_name: str,
    eval_dir: str,
    model_labels: list[str],
    n_frames: int = 5,
    frame_height: int = 200,
) -> np.ndarray | None:
    """Create a montage for one task: rows=models, cols=time steps.
    Each row has a colored header bar (LIBERO style) + frame strip."""
    rows = []

    for label in model_labels:
        video_dir = os.path.join(eval_dir, label, task_name, "videos")
        video_path, is_success = pick_best_video(video_dir)

        if video_path is None:
            print(f"  No video for {label}/{task_name}")
            continue

        frames = extract_frames_from_video(video_path, n_frames)
        if not frames:
            print(f"  Could not extract frames from {video_path}")
            continue

        # Resize all frames to same height
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            new_w = int(w * frame_height / h)
            resized.append(cv2.resize(f, (new_w, frame_height)))

        # Ensure all frames same width (use max)
        max_w = max(f.shape[1] for f in resized)
        padded = []
        for f in resized:
            if f.shape[1] < max_w:
                pad = np.zeros((frame_height, max_w - f.shape[1], 3), dtype=np.uint8)
                f = np.concatenate([f, pad], axis=1)
            padded.append(f)

        # Concatenate frames horizontally with 2px gap
        gap = np.ones((frame_height, 2, 3), dtype=np.uint8) * 40
        row_parts = []
        for i, f in enumerate(padded):
            if i > 0:
                row_parts.append(gap)
            row_parts.append(f)
        frame_strip = np.concatenate(row_parts, axis=1)

        # Create header bar
        success_rate = load_success_rate(eval_dir, label, task_name)
        header = make_header_bar(frame_strip.shape[1], label, success_rate, is_success)

        # Stack header + frame strip vertically
        model_row = np.concatenate([header, frame_strip], axis=0)
        rows.append(model_row)

    if not rows:
        return None

    # Ensure all rows same width
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded_rows.append(r)

    # Stack rows with thin separator
    sep_h = 2
    sep_color = 80
    final_parts = []
    for i, r in enumerate(padded_rows):
        if i > 0:
            sep = np.ones((sep_h, max_w, 3), dtype=np.uint8) * sep_color
            final_parts.append(sep)
        final_parts.append(r)

    return np.concatenate(final_parts, axis=0)


def load_success_rates(eval_dir: str, model_labels: list[str]) -> dict:
    """Load success rates from eval_results.json files."""
    rates = {}
    for label in model_labels:
        label_dir = os.path.join(eval_dir, label)
        if not os.path.isdir(label_dir):
            continue
        rates[label] = {}
        for task_dir in sorted(os.listdir(label_dir)):
            results_path = os.path.join(label_dir, task_dir, "eval_results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    data = json.load(f)
                rates[label][task_dir] = data["summary"]["success_rate"]
    return rates


SHORT_TASK_NAMES = {
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env": "Cuttingboard To Basket",
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env": "Bottle To Cabinet",
    "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env": "Plate To Bowl",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env": "Tray To Pot",
}


def get_short_task_name(task: str) -> str:
    if task in SHORT_TASK_NAMES:
        return SHORT_TASK_NAMES[task]
    name = task.replace("_GR1ArmsAndWaistFourierHands_Env", "")
    name = name.replace("PosttrainPnPNovelFrom", "")
    name = name.replace("SplitA", "")
    return name


def main():
    parser = argparse.ArgumentParser(description="Create evaluation montage figures")
    parser.add_argument("--eval_dir", default="outputs/eval")
    parser.add_argument("--output_dir", default="outputs/figures")
    parser.add_argument("--model_order", nargs="+", default=["base", "dpo", "rwr", "ppo"])
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--frame_height", type=int, default=180)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover tasks from eval directory
    all_tasks = set()
    for label in args.model_order:
        label_dir = os.path.join(args.eval_dir, label)
        if os.path.isdir(label_dir):
            for task_dir in os.listdir(label_dir):
                if os.path.isdir(os.path.join(label_dir, task_dir)):
                    all_tasks.add(task_dir)

    tasks = sorted(all_tasks)
    print(f"Found {len(tasks)} tasks: {tasks}")
    print(f"Models: {args.model_order}")

    # Create per-task montages
    for task in tasks:
        print(f"\nCreating montage for: {task}")
        montage = make_montage_for_task(
            task, args.eval_dir, args.model_order,
            n_frames=args.n_frames, frame_height=args.frame_height,
        )
        if montage is not None:
            short_name = get_short_task_name(task).replace(" ", "_")
            out_path = os.path.join(args.output_dir, f"montage_{short_name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path}")
        else:
            print(f"  No videos found for {task}")

    # Create combined montage (all tasks stacked)
    print("\nCreating combined montage...")
    all_montages = []
    for task in tasks:
        montage = make_montage_for_task(
            task, args.eval_dir, args.model_order,
            n_frames=args.n_frames, frame_height=args.frame_height,
        )
        if montage is not None:
            # Add task title bar
            short_name = get_short_task_name(task)
            title_h = 35
            title_bar = np.zeros((title_h, montage.shape[1], 3), dtype=np.uint8)
            title_bar[:] = (40, 40, 40)  # dark gray background
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(short_name, font, 0.75, 2)[0]
            x = (montage.shape[1] - text_size[0]) // 2
            y = (title_h + text_size[1]) // 2
            cv2.putText(title_bar, short_name, (x, y), font, 0.75,
                        (255, 255, 255), 2)
            all_montages.append(title_bar)
            all_montages.append(montage)
            # Separator
            sep = np.ones((6, montage.shape[1], 3), dtype=np.uint8) * 200
            all_montages.append(sep)

    if all_montages:
        # Ensure all same width
        max_w = max(m.shape[1] for m in all_montages)
        padded = []
        for m in all_montages:
            if m.shape[1] < max_w:
                pad = np.zeros((m.shape[0], max_w - m.shape[1], 3), dtype=np.uint8)
                m = np.concatenate([m, pad], axis=1)
            padded.append(m)

        combined = np.concatenate(padded, axis=0)
        out_path = os.path.join(args.output_dir, "combined_montage.png")
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Combined montage saved: {out_path}")

    print("\nDone! Figures in:", args.output_dir)


if __name__ == "__main__":
    main()

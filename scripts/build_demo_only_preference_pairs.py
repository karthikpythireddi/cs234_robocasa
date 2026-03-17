#!/usr/bin/env python3
"""
build_demo_only_preference_pairs.py

Builds preference pairs directly from demo data — NO simulation required.

  Winners: Full expert demo trajectories (complete task)
  Losers:  Truncated demo trajectories (cut at 20-50% — task not completed)

This avoids the MuJoCo rendering issues on headless environments while
producing clean preference signal: full demos clearly outperform partial ones.

Usage (from Isaac-GR00T root):
    python scripts/build_demo_only_preference_pairs.py \
        --task_name PosttrainPnPNovelFromCuttingboardToBasketSplitA \
        --demo_data_dir examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA \
        --n_pairs 50 \
        --output_dir preference_data/gr1_demo_pairs
"""

import argparse
import json
import os
import random

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq


# ── GR1 embodiment config ─────────────────────────────────────────────────

STATE_SLICES = {
    "left_arm":  (0, 7),
    "right_arm": (22, 29),
    "left_hand": (7, 13),
    "right_hand": (29, 35),
    "waist":     (41, 44),
}
ACTION_SLICES = STATE_SLICES

ACTION_HORIZON = 16
N_ACTION_STEPS = 8

# Truncate losers at this fraction of the full episode length
TRUNCATE_MIN = 0.2
TRUNCATE_MAX = 0.5


# ── Data loading ──────────────────────────────────────────────────────────

def get_available_episodes(data_dir: str) -> list:
    episodes_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            episodes.append(ep["episode_index"])
    return episodes


def _decode_video(video_path: str, expected_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    arr = np.stack(frames)
    if len(arr) > expected_frames:
        arr = arr[:expected_frames]
    elif len(arr) < expected_frames:
        pad = np.repeat(arr[-1:], expected_frames - len(arr), axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def _chunk_actions(per_step: np.ndarray, horizon: int) -> np.ndarray:
    T, D = per_step.shape
    chunk_indices = list(range(0, T, N_ACTION_STEPS))
    chunks = []
    for t in chunk_indices:
        window = per_step[t:t + horizon]
        if len(window) < horizon:
            pad = np.repeat(window[-1:], horizon - len(window), axis=0)
            window = np.concatenate([window, pad], axis=0)
        chunks.append(window)
    return np.array(chunks)


def load_demo_episode(data_dir: str, episode_idx: int) -> dict:
    chunk_idx = episode_idx // 1000
    parquet_path = os.path.join(
        data_dir, "data", f"chunk-{chunk_idx:03d}",
        f"episode_{episode_idx:06d}.parquet"
    )
    video_path = os.path.join(
        data_dir, "videos", f"chunk-{chunk_idx:03d}",
        "observation.images.ego_view",
        f"episode_{episode_idx:06d}.mp4"
    )

    df = pq.read_table(parquet_path).to_pandas()
    T = len(df)

    state_vec = np.stack(df["observation.state"].values).astype(np.float32)
    obs = {}
    for key, (start, end) in STATE_SLICES.items():
        obs[f"state.{key}"] = state_vec[:, start:end]

    frames = _decode_video(video_path, T)
    obs["video.ego_view"] = frames

    action_vec = np.stack(df["action"].values).astype(np.float32)
    actions = {}
    for key, (start, end) in ACTION_SLICES.items():
        per_step = action_vec[:, start:end]
        actions[f"action.{key}"] = _chunk_actions(per_step, ACTION_HORIZON)

    T_chunks = actions[f"action.{list(ACTION_SLICES.keys())[0]}"].shape[0]
    for k in obs:
        obs[k] = obs[k][:T_chunks]

    return {"obs": obs, "actions": actions, "success": True,
            "cumulative_reward": 1.0, "length": T_chunks}


def truncate_episode(episode: dict, frac: float) -> dict:
    """Create a loser by truncating the episode at frac of its length."""
    T = episode["length"]
    cut = max(1, int(T * frac))

    obs = {k: v[:cut] for k, v in episode["obs"].items()}
    actions = {k: v[:cut] for k, v in episode["actions"].items()}

    return {"obs": obs, "actions": actions, "success": False,
            "cumulative_reward": frac, "length": cut}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--demo_data_dir", required=True)
    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--output_dir", default="preference_data/gr1_demo_pairs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"Demo-Only Preference Pair Builder (no simulation)")
    print(f"{'='*60}")
    print(f"Task         : {args.task_name}")
    print(f"Demo data    : {args.demo_data_dir}")
    print(f"n_pairs      : {args.n_pairs}")
    print(f"Losers       : truncated at {int(TRUNCATE_MIN*100)}-{int(TRUNCATE_MAX*100)}% of episode")
    print(f"{'='*60}\n")

    episodes = get_available_episodes(args.demo_data_dir)
    if len(episodes) < args.n_pairs:
        print(f"[warn] Only {len(episodes)} episodes available, using all.")
    selected = random.sample(episodes, min(args.n_pairs, len(episodes)))

    os.makedirs(args.output_dir, exist_ok=True)
    task_env_name = f"{args.task_name}_GR1ArmsAndWaistFourierHands_Env"
    output_path = os.path.join(args.output_dir, f"{task_env_name}_demo_preferences.hdf5")

    pairs = []
    for i, ep_idx in enumerate(selected):
        print(f"  Pair {i+1}/{len(selected)}: episode {ep_idx} ...", end=" ", flush=True)
        try:
            winner = load_demo_episode(args.demo_data_dir, ep_idx)
            frac = random.uniform(TRUNCATE_MIN, TRUNCATE_MAX)
            loser = truncate_episode(winner, frac)
            pairs.append((winner, loser))
            print(f"len={winner['length']} chunks, loser cut={frac:.0%}")
        except Exception as e:
            print(f"SKIP ({e})")

    print(f"\nSaving {len(pairs)} pairs to {output_path} ...")
    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["n_pairs"] = len(pairs)
        meta.attrs["task_name"] = task_env_name

        for i, (winner, loser) in enumerate(pairs):
            grp = f.create_group(f"pair_{i}")
            grp.attrs["winner_success"] = winner["success"]
            grp.attrs["loser_success"] = loser["success"]
            grp.attrs["winner_cumulative_reward"] = winner["cumulative_reward"]
            grp.attrs["loser_cumulative_reward"] = loser["cumulative_reward"]
            grp.attrs["preference_type"] = "demo_vs_truncated"

            for label, traj in [("winner", winner), ("loser", loser)]:
                t_grp = grp.create_group(label)
                act_grp = t_grp.create_group("actions")
                for ak, av in traj["actions"].items():
                    if av is not None and len(av) > 0:
                        act_grp.create_dataset(ak, data=av)
                obs_grp = t_grp.create_group("obs")
                for ok, ov in traj["obs"].items():
                    if ov is not None and len(ov) > 0:
                        obs_grp.create_dataset(ok, data=ov)

            if (i + 1) % 10 == 0:
                print(f"    Saved {i+1}/{len(pairs)} pairs...")

    print(f"\n[done] {len(pairs)} pairs saved to {output_path}")


if __name__ == "__main__":
    main()

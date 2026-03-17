#!/usr/bin/env python3
"""
generate_report_figures.py

Generate publication-quality figures for the CS234 final report from GR00T eval results.

Produces:
  1. Bar chart: Success rate by task and model
  2. Bar chart: Average success rate across tasks
  3. Table image: Full results table
  4. Episode length distribution (box plot)

Usage:
  python scripts/generate_report_figures.py --eval_dir outputs/eval --output_dir report_figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SHORT_NAMES = {
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env": "Cuttingboard\nTo Basket",
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env": "Bottle To\nCabinet",
    "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env": "Plate\nTo Bowl",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env": "Tray\nTo Pot",
}

TASK_ORDER = [
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
]

MODEL_ORDER = ["base", "dpo", "rwr", "ppo"]
MODEL_LABELS = {"base": "Base (GR00T N1.6)", "dpo": "DPO", "rwr": "RLHF", "ppo": "PPO"}
MODEL_COLORS = {"base": "#4C72B0", "dpo": "#DD8452", "rwr": "#55A868", "ppo": "#C44E52"}


def load_all_results(eval_dir):
    """Load all eval_results.json into {model: {task: data}}."""
    results = {}
    for model in MODEL_ORDER:
        model_dir = os.path.join(eval_dir, model)
        if not os.path.isdir(model_dir):
            continue
        results[model] = {}
        for task in os.listdir(model_dir):
            fp = os.path.join(model_dir, task, "eval_results.json")
            if os.path.isfile(fp):
                with open(fp) as f:
                    results[model][task] = json.load(f)
    return results


def fig_success_rate_by_task(results, output_dir):
    """Grouped bar chart: success rate per task, grouped by model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_tasks = len(TASK_ORDER)
    n_models = len(MODEL_ORDER)
    bar_width = 0.18
    x = np.arange(n_tasks)

    for i, model in enumerate(MODEL_ORDER):
        rates = []
        for task in TASK_ORDER:
            if task in results.get(model, {}):
                sr = results[model][task]["summary"]["success_rate"] * 100
            else:
                sr = 0
            rates.append(sr)
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, rates, bar_width,
                      label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                      edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, rates):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("GR00T N1.6: Preference Fine-Tuning — Success Rate by Task (20 eps)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[t] for t in TASK_ORDER], fontsize=9)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "success_rate_by_task.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_average_success_rate(results, output_dir):
    """Single bar chart: average success rate across all tasks per model."""
    fig, ax = plt.subplots(figsize=(6, 4))

    avgs = []
    for model in MODEL_ORDER:
        rates = []
        for task in TASK_ORDER:
            if task in results.get(model, {}):
                rates.append(results[model][task]["summary"]["success_rate"] * 100)
        avgs.append(np.mean(rates) if rates else 0)

    bars = ax.bar(
        [MODEL_LABELS[m] for m in MODEL_ORDER], avgs,
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        edgecolor="white", linewidth=0.5, width=0.5
    )
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Average Success Rate (%)", fontsize=12)
    ax.set_title("GR00T N1.6: Preference Fine-Tuning: Average Success Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "average_success_rate.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_episode_lengths(results, output_dir):
    """Box plot of episode lengths per model (all tasks combined)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    data = []
    labels = []
    for model in MODEL_ORDER:
        lengths = []
        for task in TASK_ORDER:
            if task in results.get(model, {}):
                for ep in results[model][task].get("episodes", []):
                    lengths.append(ep["length"])
        data.append(lengths)
        labels.append(MODEL_LABELS[model])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.7)

    ax.set_ylabel("Episode Length (steps)", fontsize=12)
    ax.set_title("Episode Length Distribution by Model", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "episode_lengths.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_results_table(results, output_dir):
    """Render a clean results table as an image."""
    task_short = {
        "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env": "Cuttingboard To Basket",
        "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env": "Bottle To Cabinet",
        "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env": "Plate To Bowl",
        "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env": "Tray To Pot",
    }

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")

    col_labels = ["Task"] + [MODEL_LABELS[m] for m in MODEL_ORDER]
    table_data = []

    for task in TASK_ORDER:
        row = [task_short[task]]
        for model in MODEL_ORDER:
            if task in results.get(model, {}):
                s = results[model][task]["summary"]
                row.append(f"{s['n_success']}/{s['n_episodes']} ({s['success_rate']*100:.0f}%)")
            else:
                row.append("N/A")
        table_data.append(row)

    # Average row
    avg_row = ["AVERAGE"]
    for model in MODEL_ORDER:
        rates = []
        for task in TASK_ORDER:
            if task in results.get(model, {}):
                rates.append(results[model][task]["summary"]["success_rate"] * 100)
        avg_row.append(f"{np.mean(rates):.1f}%" if rates else "N/A")
    table_data.append(avg_row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header
    for j, label in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")

    # Style average row
    for j in range(len(col_labels)):
        cell = table[len(table_data), j]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(fontweight="bold")

    # Highlight best per task (skip avg row)
    for i, task in enumerate(TASK_ORDER):
        best_val = -1
        best_j = -1
        for j, model in enumerate(MODEL_ORDER):
            if task in results.get(model, {}):
                sr = results[model][task]["summary"]["success_rate"]
                if sr > best_val:
                    best_val = sr
                    best_j = j + 1  # +1 for task column
        if best_j > 0 and best_val > 0:
            table[i + 1, best_j].set_text_props(fontweight="bold", color="#2E7D32")

    ax.set_title("GR00T N1.6: Preference Fine-Tuning Evaluation Results (20 episodes per task)",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "results_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="outputs/eval")
    parser.add_argument("--output_dir", default="report_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_all_results(args.eval_dir)

    if not results:
        print("No results found!")
        return

    fig_success_rate_by_task(results, args.output_dir)
    fig_average_success_rate(results, args.output_dir)
    fig_episode_lengths(results, args.output_dir)
    fig_results_table(results, args.output_dir)

    print(f"\nAll figures saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()

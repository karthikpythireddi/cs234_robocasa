# CS234: Preference Optimization for Humanoid Robotic Control

**Stanford CS234: Reinforcement Learning — Final Project**
**Karthik Pythireddi**

Applying preference-based fine-tuning (DPO, RWR, PPO) to NVIDIA's GR00T N1.6-3B Vision-Language-Action model on RoboCasa tabletop manipulation tasks using the GR1 humanoid robot.

---

## Overview

This project investigates whether preference optimization algorithms can improve a pre-trained VLA model (GR00T N1.6-3B) beyond supervised fine-tuning (SFT), using human preference data collected from simulation rollouts.

**Pipeline:**
```
SFT Fine-tuning → Rollout Collection → Preference Pair Construction → DPO / RWR / PPO Training → Evaluation
```

---

## Tasks

Four RoboCasa tabletop manipulation tasks using the GR1 humanoid robot:

| Task | Description |
|------|-------------|
| `PnPBottleToCabinetClose` | Pick bottle and place in cabinet, then close |
| `PosttrainPnPNovelFromCuttingboardToBasketSplitA` | Pick item from cutting board and place in basket |
| `PosttrainPnPNovelFromPlateToBowlSplitA` | Pick item from plate and place in bowl |
| `PosttrainPnPNovelFromTrayToPotSplitA` | Pick item from tray and place in pot |

---

## Results

Success rate over 20 episodes per task:

| Task | Base SFT | DPO | RWR | PPO |
|------|----------|-----|-----|-----|
| Bottle → Cabinet | 50% | 0% | 0% | 15% |
| Cutting Board → Basket | 65% | 0% | 0% | 0% |
| Plate → Bowl | 40% | 0% | 0% | 0% |
| Tray → Pot | 55% | 0% | 5% | 0% |
| **Average** | **52.5%** | **0%** | **1.25%** | **3.75%** |

**Key finding:** DPO and RWR suffered catastrophic forgetting — fine-tuning on preference pairs (all demo-vs-failure) with lr=1e-5 for 3 epochs on the full model erased the base SFT policy. This is a known challenge when applying preference optimization to large VLA models with flow-matching action heads.

---

## Repository Structure

```
├── gr00t_rlhf/
│   └── algos/
│       ├── dpo.py              # DPO training with flow-matching proxy loss
│       ├── rwr.py              # Reward-Weighted Regression
│       └── ppo.py              # PPO fine-tuning
├── scripts/
│   ├── build_demo_preference_pairs.py   # Collect preference pairs from rollouts
│   ├── eval_policy.py                   # Evaluate policy via GR00T inference server
│   ├── make_linkedin_video.py           # Generate side-by-side comparison videos
│   ├── make_eval_montage.py             # Montage of eval episodes
│   ├── print_eval_summary.py            # Print success rate tables
│   ├── generate_report_figures.py       # Generate paper figures
│   └── finetune_gr1_h100.sh             # SFT fine-tuning launch script
├── report_figures/                      # Figures used in the paper
└── visualizations/                      # Preference pair visualizations
```

---

## Setup

Requires [Isaac-GR00T](https://github.com/NVIDIA-Cosmos/Isaac-GR00T) and [RoboCasa](https://github.com/robocasa/robocasa) installed.

```bash
# Run SFT fine-tuning
bash scripts/finetune_gr1_h100.sh

# Collect preference pairs (requires GR00T inference server running)
python scripts/build_demo_preference_pairs.py --task PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env

# Train DPO
python gr00t_rlhf/algos/dpo.py --dataset_path outputs/preferences --output_dir outputs/dpo

# Train RWR
python gr00t_rlhf/algos/rwr.py --dataset_path outputs/preferences --output_dir outputs/rwr

# Evaluate
python scripts/eval_policy.py --task PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env --label base

# Generate comparison video
python scripts/make_linkedin_video.py --eval_dir outputs/eval --combined
```

---

## Model

- **Base model:** [NVIDIA GR00T N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- **Action representation:** Flow-matching (continuous diffusion-based)
- **DPO proxy loss:** Denoising MSE used as log-probability surrogate
- **Preference data:** 200 pairs total (50 per task), demo-vs-failure type
- **Hardware:** NVIDIA H100 80GB (Lightning AI)

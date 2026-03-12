# Risk-Aware Active VLA for Autonomous Driving

Full implementation of a risk-aware autonomous driving system with:
- Self-supervised visual adaptation (DINOv2)
- Structured latent risk vector
- Multimodal latent fusion
- CVaR-aware reinforcement learning
- Active peeking under occlusion
- Grounded LLM explanations

## Architecture

```
Camera Frames
    ↓
[SSL-adapted DINOv2]  [YOLO Tracker]  [DepthAnything]  [Lane Detector]  [PIP Model]
    ↓                      ↓                ↓                 ↓               ↓
    h_t              detections_t       depth_t           lane_t         p_cross
    └──────────────────────┴────────────────┴─────────────────┴───────────────┘
                                     ↓
                          [Fusion Head g_ω]
                                     ↓
            Ẑ_t = [ol, oc, or, pcoll, TTC, pcross, dped, v, elane, ψ̇]
                    ↙                                        ↘
         [PPO Policy π_φ]                            [Frozen LLM]
              ↓                                           ↓
        a_t = [steer, throttle, brake]         "Braking due to high occlusion..."
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install CARLA
```bash
# Download CARLA 0.9.15 from https://carla.org/
# Then install the Python API:
pip install carla==0.9.15
```

### 3. Run Stage 0 (Setup Verification)
```bash
python scripts/stage0_setup.py
```

### 4. Run All Stages
```bash
# Full training pipeline
python train.py --all

# Or run specific stages
python train.py --stages 0 1 2 3

# Train with CVaR (Stage 8)
python train.py --stages 8 --cvar

# Quick LLM test without GPU-heavy model
python scripts/stage11_llm.py --fallback
```

## Project Structure

```
risk_aware_vla/
├── config/
│   └── config.yaml              ← Master configuration
├── perception/
│   ├── backbone.py              ← DINOv2 ViT-B/14 + SSL head
│   ├── detector.py              ← YOLOv8 + ByteTrack
│   ├── depth.py                 ← Depth-Anything V2
│   ├── lane.py                  ← Ultra-fast lane detector
│   └── pedestrian_intention.py  ← PIP model + PIE/JAAD pipeline
├── latent/
│   ├── labels.py                ← Ground-truth latent label computer
│   └── fusion.py                ← Multimodal fusion head
├── ssl/
│   └── ssl_trainer.py           ← VICReg-style SSL adaptation
├── rl/
│   ├── env.py                   ← CARLA Gymnasium environment
│   └── ppo.py                   ← PPO + CVaRCallback
├── llm/
│   └── explainer.py             ← Frozen LLM explanation branch
├── eval/
│   └── evaluator.py             ← Full evaluation suite
├── scripts/
│   ├── stage0_setup.py          ← Environment setup
│   ├── stage1_pie_pipeline.py   ← PIP model training
│   ├── stage3_latent_labels.py  ← Label generation check
│   ├── stage4_ssl.py            ← SSL training
│   ├── stage5_fusion.py         ← Fusion head training
│   ├── stage7_ppo.py            ← PPO training
│   ├── stage10_zhat.py          ← Switch to predicted Ẑ
│   ├── stage11_llm.py           ← LLM explainer test
│   └── run_final_eval.py        ← Final evaluation
└── train.py                     ← Master orchestrator
```

## Latent Vector

| Index | Name    | Description                   | Loss |
|-------|---------|-------------------------------|------|
| 0     | ol      | Left sector occlusion         | BCE  |
| 1     | oc      | Center sector occlusion       | BCE  |
| 2     | or      | Right sector occlusion        | BCE  |
| 3     | pcoll   | Short-horizon collision prob  | BCE  |
| 4     | TTC     | Time-to-collision (norm.)     | MSE  |
| 5     | pcross  | Pedestrian crossing prob      | BCE  |
| 6     | dped    | Distance to nearest ped (norm.) | MSE |
| 7     | v       | Ego speed (norm.)             | MSE  |
| 8     | elane   | Lane-center error (norm.)     | MSE  |
| 9     | yaw_rate| Yaw rate (norm.)              | MSE  |

## Stage-by-Stage Guide

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | stage0_setup.py | Install check, folder structure, CARLA logger |
| 1 | stage1_pie_pipeline.py | PIE data + PIP model training |
| 2 | stage2_perception.py | Perception stack verification |
| 3 | stage3_latent_labels.py | Latent label sanity checks |
| 4 | stage4_ssl.py | SSL backbone adaptation |
| 5 | stage5_fusion.py | Fusion head training |
| 6 | stage6_domain_rand.py | Domain randomization config |
| 7 | stage7_ppo.py | PPO on true Z |
| 8 | stage7_ppo.py --cvar | CVaR tail-risk training |
| 9 | stage9_peeking.py | Active peeking reward check |
| 10 | stage10_zhat.py | Switch to predicted Ẑ |
| 11 | stage11_llm.py | LLM explanation branch |
| 12 | run_final_eval.py | Final evaluation across all scenes |

## Key Papers / Models

- **DINOv2**: Oquab et al., 2023
- **VICReg** (SSL): Bardes et al., 2022
- **Depth-Anything**: Yang et al., 2024
- **YOLOv8**: Ultralytics, 2023
- **PIP**: Rasouli et al., 2019 (PIE dataset)
- **CVaR-RL**: Tamar et al., 2015; Chow & Ghavamzadeh, 2014
- **PPO**: Schulman et al., 2017

## Datasets

| Dataset | Purpose | Link |
|---------|---------|------|
| PIE | Pedestrian intention training | nvision2.eecs.yorku.ca |
| JAAD | Pedestrian intention validation | nvision2.eecs.yorku.ca |
| CARLA | RL environment + latent labels | carla.org |

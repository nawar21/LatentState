"""
Master Training Orchestrator
Runs all stages in sequence, or a specific stage subset.

Usage:
  python train.py --all                         # run all stages
  python train.py --stages 0 1 2 3              # run specific stages
  python train.py --stages 7 --cvar             # PPO with CVaR
  python train.py --stages 4 --frames data/carla/frames

Stages:
  0  - Environment setup & verification
  1  - PIE pedestrian pipeline
  2  - Perception stack integration check
  3  - Latent label generation in CARLA
  4  - SSL backbone adaptation
  5  - Latent fusion head training
  6  - Domain randomization
  7  - PPO on true Z
  8  - CVaR tail-risk training
  9  - Active peeking
  10 - Replace Z with Zhat
  11 - LLM explainer
  12 - Final evaluation
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGE_SCRIPTS = {
    0:  ["python", "scripts/stage0_setup.py"],
    1:  ["python", "scripts/stage1_pie_pipeline.py"],
    2:  ["python", "scripts/stage2_perception.py"],
    3:  ["python", "scripts/stage3_latent_labels.py"],
    4:  ["python", "scripts/stage4_ssl.py"],
    5:  ["python", "scripts/stage5_fusion.py"],
    6:  ["python", "scripts/stage6_domain_rand.py"],
    7:  ["python", "scripts/stage7_ppo.py"],
    8:  ["python", "scripts/stage7_ppo.py", "--cvar"],
    9:  ["python", "scripts/stage9_peeking.py"],
    10: ["python", "scripts/stage10_zhat.py"],
    11: ["python", "scripts/stage11_llm.py"],
    12: ["python", "scripts/run_final_eval.py"],
}

STAGE_NAMES = {
    0:  "Environment Setup",
    1:  "PIE Pedestrian Pipeline",
    2:  "Frozen Perception Stack",
    3:  "Latent Label Generation",
    4:  "SSL Visual Adaptation",
    5:  "Latent Fusion Head",
    6:  "Domain Randomization",
    7:  "PPO on True Z",
    8:  "CVaR Tail-Risk Training",
    9:  "Active Peeking",
    10: "Replace Z with Zhat",
    11: "LLM Explainer",
    12: "Final Evaluation",
}


def run_stage(stage_id: int, extra_args: list = []) -> bool:
    name = STAGE_NAMES.get(stage_id, f"Stage {stage_id}")
    cmd = STAGE_SCRIPTS.get(stage_id, [])
    if not cmd:
        print(f"[train.py] No script defined for stage {stage_id}")
        return False

    full_cmd = cmd + extra_args
    print(f"\n{'='*65}")
    print(f"  Stage {stage_id}: {name}")
    print(f"  Command: {' '.join(full_cmd)}")
    print(f"{'='*65}\n")

    result = subprocess.run(full_cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\n[train.py] Stage {stage_id} FAILED (exit code {result.returncode})")
        return False
    print(f"\n[train.py] Stage {stage_id}: {name} — DONE ✓")
    return True


def main():
    parser = argparse.ArgumentParser(description="Risk-Aware VLA Master Trainer")
    parser.add_argument("--all", action="store_true", help="Run all stages 0–12")
    parser.add_argument("--stages", nargs="+", type=int, help="Specific stage IDs to run")
    parser.add_argument("--cvar", action="store_true", help="Enable CVaR (stage 7→8)")
    parser.add_argument("--fallback", action="store_true", help="LLM fallback only")
    parser.add_argument("--frames_dir", default="data/carla/frames")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        stages = list(range(13))
    elif args.stages:
        stages = args.stages
    else:
        parser.print_help()
        sys.exit(0)

    failed = []
    for s in stages:
        extra = []
        if s == 4 and args.frames_dir:
            extra = ["--frames_dir", args.frames_dir]
        if s == 7 and args.cvar:
            extra = ["--cvar"]
        if s == 7 and args.timesteps:
            extra += ["--timesteps", str(args.timesteps)]
        if s == 11 and args.fallback:
            extra = ["--fallback"]

        ok = run_stage(s, extra)
        if not ok:
            failed.append(s)

    print(f"\n{'='*65}")
    if failed:
        print(f"  Training complete with failures in stages: {failed}")
    else:
        print(f"  All stages completed successfully ✓")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

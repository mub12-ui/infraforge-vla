"""
InfraForge VLA — Robustness Evaluation (10 randomized seeds)
"""

import json
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scripted_controller import run_instruction, BOX_START

N_SEEDS = 10
SUCCESS_THRESHOLD = 2
X_RANGE = (-0.025, 0.015)
Y_RANGE = (-0.015, 0.025)
INSTRUCTION = "pick up the mug with arm A"


def run_evaluation(n_seeds=N_SEEDS, seed_base=42):
    results = []
    for i in range(n_seeds):
        rng = random.Random(seed_base + i)
        dx = rng.uniform(*X_RANGE)
        dy = rng.uniform(*Y_RANGE)
        box_pos = [BOX_START[0] + dx, BOX_START[1] + dy, BOX_START[2]]

        r = run_instruction(INSTRUCTION, verbose=False, box_pos=box_pos)
        success = r["max_reward"] >= SUCCESS_THRESHOLD

        results.append({
            "seed": seed_base + i, "offset_x": round(dx, 4), "offset_y": round(dy, 4),
            "box_pos": box_pos, "max_reward": r["max_reward"], "success": success,
        })
        print(f"Seed {seed_base+i:3d}  offset=({dx:+.4f},{dy:+.4f})  "
              f"max_reward={r['max_reward']}  {'PASS' if success else 'FAIL'}")

    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'='*50}\nRESULT: {n_success}/{n_seeds} seeds succeeded ({100*n_success/n_seeds:.0f}%)\n{'='*50}")

    return {"instruction": INSTRUCTION, "n_seeds": n_seeds, "success_threshold": SUCCESS_THRESHOLD,
            "randomization_range": {"x": X_RANGE, "y": Y_RANGE}, "results": results,
            "n_success": n_success, "success_rate": n_success / n_seeds}


if __name__ == "__main__":
    summary = run_evaluation()
    with open("robustness_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote robustness_results.json")

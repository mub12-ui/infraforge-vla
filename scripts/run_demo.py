"""
InfraForge VLA — Demo Entry Point

Run the full pipeline end-to-end: a natural-language instruction goes
in, gets parsed, drives the simulated bimanual arms, and reports the
resulting reward/success trace.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --instruction "pick up the plate with arm A, hand it to arm B"
"""

import argparse
import sys
import os

# Allow running this script directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scripted_controller import run_instruction  # noqa: E402

DEFAULT_INSTRUCTION = "pick up the mug with arm A, hand it to arm B"


def main():
    parser = argparse.ArgumentParser(description="Run the InfraForge VLA demo pipeline")
    parser.add_argument(
        "--instruction", type=str, default=DEFAULT_INSTRUCTION,
        help="Natural-language command to execute (default: a pick-up + hand-off example)",
    )
    args = parser.parse_args()

    result = run_instruction(args.instruction, verbose=True)

    print("\n" + "=" * 50)
    print(f"RESULT: max_reward={result['max_reward']}/4  "
          f"({'SUCCESS' if result['max_reward'] >= 4 else 'partial / in progress'})")
    print("=" * 50)


if __name__ == "__main__":
    main()

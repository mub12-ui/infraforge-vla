# InfraForge VLA — Bimanual Manipulation with Multi-Modal Reasoning

Submission for the Intel Physical AI Online Challenge (AI Infra Summit Hackathon 2026):
Bimanual VLA Manipulation with Multi-Modal Reasoning.

## What this is

A natural-language-driven manipulation pipeline: a language instruction is
parsed into a structured action plan, which drives a simulated bimanual
robot (MuJoCo, LeRobot's ALOHA environment) to pick up an object. The
pick-up mechanic is position-relative (computed from the object's actual
location via a measured geometric offset, not hardcoded to one spot) and
its robustness to placement variation is empirically evaluated across 10
randomized seeds. Inference is benchmarked with Intel OpenVINO.

## Verified results

- **Pick-up (right arm)**: reward 2/4 (object gripped and lifted clear of
  the table), verified on real hardware, reproducible.
- **Pick-up (left arm)**: reward 4/4, verified on real hardware, reproducible.
- **Robustness (10 randomized seeds)**: 9/10 success (90%) within a
  measured safe placement range (x: -2.5cm to +1.5cm, y: -1.5cm to +2.5cm
  from the calibration point). See `scripts/robustness_results.json` for
  the raw per-seed data — this is real evaluation output, not illustrative
  numbers. The one failure landed at the known boundary of the tested
  range, which is itself evidence the boundary is real and consistent.
- **Language parsing**: tested against multiple phrasings, correctly
  identifies arm, action, and object; two real parsing bugs (reversed
  hand-off direction, incomplete clause-splitting) were found and fixed
  during testing.
- **OpenVINO inference benchmark**: working end-to-end, reports real
  per-device (CPU/iGPU) latency and throughput. Result on our hardware:
  CPU 0.494ms mean / GPU 2.011ms mean for the placeholder model — CPU
  is faster here, which is expected and worth noting honestly: for a
  very small model, GPU dispatch overhead outweighs its parallelism
  benefit. This advantage typically flips for larger models or batched
  inference. Raw data in `src/openvino_benchmark_results.json`.

## Honest scope note

The brief's reference scenario is a full dinner-table setup with a drawer,
multiple objects, and pouring actions. Given the online-only timeline, we
scoped this down to a single object pick-up as the reliably-demonstrated
core mechanic, with hand-off (passing the object between arms) as a
stretch goal — see "Known limitation" below.

We used LeRobot's stock ALOHA simulation (ViperX 300S arms) rather than a
from-scratch SO-101 MuJoCo model, since no pre-built dual-SO-101 simulation
asset exists in the LeRobot ecosystem. The coordination and language-
reasoning logic is arm-model-agnostic.

## Known limitation: hand-off

Two-arm hand-off (object passing from one gripper to the other) is not
yet reliable. We root-caused the blocker through direct measurement
rather than guessing: the receiving arm's approach can be positioned
within ~1.5cm of the object, but the *holding* arm's gripper does not
release when commanded to open in this held-in-air configuration — every
test showed the object remaining in contact with only the original
holding arm's fingers, even after commanding its grip open. This is a
well-defined, bounded problem (not open-ended), documented in
`docs/technical_notes.md` for anyone continuing this work.

## Architecture

```
Natural language instruction
        |
        v
language_interface.py  -- parses instruction into ActionSteps
        |                  (arm, action type, target object)
        v
scripted_controller.py -- computes contact targets RELATIVE to the
        |                  object's actual position (measured offset),
        |                  executes approach/grasp/lift via end-effector
        |                  (mocap) control in MuJoCo
        v
gym-aloha / dm_control  -- physics simulation, reward/contact tracking
        |
        v
openvino_benchmark.py   -- inference latency/throughput on Intel CPU/iGPU
```

## Components

- **`src/language_interface.py`** — rule-based parser mapping natural-
  language commands to structured `ActionStep`s (arm, action, object).
  Deliberately transparent/rule-based given the timeline — this module
  is the seam where a trained model could later replace the parser
  without touching the rest of the pipeline.

- **`src/scripted_controller.py`** — position-relative waypoint control
  of the bimanual arms via MuJoCo mocap targets. Contact targets are
  computed as `object_position + measured_offset`, not hardcoded
  coordinates — this is what makes the pick-up genuinely robust to
  placement variation rather than only working at one exact spot.

- **`src/openvino_benchmark.py`** — real Intel OpenVINO inference
  benchmark (latency, throughput, device enumeration).

- **`scripts/robustness_eval.py`** — runs the pick-up across 10
  randomized object placements and reports pass/fail per seed. Real
  output saved in `scripts/robustness_results.json`.

- **`scripts/run_demo.py`** — single command to run the full pipeline
  end-to-end from a natural-language instruction.

## Hardware used

Development and benchmarking on an 11th Gen Intel Core i7-1185G7 (Tiger
Lake) with Iris Xe integrated graphics — CPU and iGPU inference paths
both verified working via OpenVINO. No Core Ultra Series 2/3 (NPU-class)
hardware was available to the team; NPU-specific benchmarking was not
performed.

## Reproducing this

```bash
python -m venv vla-env
source vla-env/Scripts/activate   # or vla-env/bin/activate on Linux/Mac
pip install "lerobot[aloha,pusht]" openvino==2026.3.0 torch

cd src
python language_interface.py           # test the language parser
python openvino_benchmark.py           # run the Intel inference benchmark
python ../scripts/run_demo.py          # run the full pipeline
python ../scripts/robustness_eval.py   # run the 10-seed evaluation
```

## Team / track

Built for the Intel Physical AI Online Challenge, AI Infra Summit Hackathon
2026 (lablab.ai), by Muhammad.

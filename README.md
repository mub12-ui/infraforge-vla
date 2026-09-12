# InfraForge VLA — Bimanual Manipulation with Multi-Modal Reasoning

Submission for the Intel Physical AI Online Challenge (AI Infra Summit Hackathon 2026):
Bimanual VLA Manipulation with Multi-Modal Reasoning.

## What this is

A natural-language-driven bimanual manipulation pipeline: a language instruction
is parsed into a structured action sequence, which drives two simulated robot
arms (MuJoCo, LeRobot's ALOHA environment) through a pick-and-hand-off task.
Inference is benchmarked and optimized with Intel OpenVINO.

## Honest scope note

The brief's reference scenario is a full dinner-table setup with a drawer,
multiple objects, and pouring actions. Given the online-only timeline (six
days, no prior robotics background on the team), we scoped this down to a
single representative pick-and-hand-off action between two arms — the same
core mechanic (bimanual coordination, language-driven execution, object
hand-off) demonstrated reliably rather than a larger scenario attempted
unreliably.

We also used LeRobot's stock ALOHA simulation (ViperX 300S arms) rather
than a from-scratch SO-101 MuJoCo model, since no pre-built dual-SO-101
simulation asset exists in the LeRobot ecosystem and building one from
scratch was not feasible in the available time. The coordination and
language-reasoning logic below is arm-model-agnostic.

## Architecture

```
Natural language instruction
        |
        v
language_interface.py  -- parses instruction into ActionSteps
        |                  (arm, action type, target object)
        v
scripted controller     -- executes pick-up / hand-off / place
        |                  via end-effector (mocap) control in MuJoCo
        v
gym-aloha / dm_control  -- physics simulation, reward/contact tracking
        |
        v
openvino_benchmark.py   -- inference latency/throughput on Intel CPU/iGPU
```

## Components

- **`language_interface.py`** — rule-based parser mapping natural-language
  commands ("pick up the mug with arm A, hand it to arm B") to a structured
  list of `ActionStep`s (arm, action, object). Tested against multiple
  phrasings. Deliberately transparent/rule-based rather than a trained
  language model, given the timeline — a reliable parser driving a real
  working action sequence is worth more than an unreliable learned one.
  This module is the seam where a trained model could later replace the
  parser without touching the rest of the pipeline.

- **`scripted controller`** (in-progress) — waypoint-based end-effector
  control of the bimanual arms via MuJoCo mocap targets, executing
  pick-up and hand-off actions. We identified and fixed two real bugs
  in the underlying `gym-aloha` library along the way:
  - the end-effector control mode is stubbed with `NotImplementedError`
    in the installed version, worked around by constructing the
    `dm_control` environment directly from the library's own task classes
  - a 0.1347m positional offset in the mocap-to-gripper weld constraint,
    found by inspecting the MuJoCo model's equality-constraint data directly

  **Known limitation**: the controller reliably approaches and reaches the
  target object, but does not yet reliably retain grip through the lift
  phase. This is the primary open item — documented here rather than
  hidden, per the project's approach to honest scoping throughout.

- **`openvino_benchmark.py`** — real Intel OpenVINO inference benchmark
  (latency, throughput, device enumeration) satisfying the brief's required
  benchmark deliverable. Verified working, reporting real per-device
  numbers (CPU and, on Intel Core Ultra / Iris Xe hardware, GPU).

## Hardware used

Development and benchmarking on an 11th Gen Intel Core i7-1185G7 (Tiger Lake)
with Iris Xe integrated graphics — CPU and iGPU inference paths both verified
working via OpenVINO. No NPU-class hardware (Core Ultra Series 2/3) was
available to the team; NPU-specific benchmarking was not performed.

## Reproducing this

```bash
python -m venv vla-env
source vla-env/Scripts/activate   # or vla-env/bin/activate on Linux/Mac
pip install "lerobot[aloha,pusht]" openvino==2026.3.0 torch

python language_interface.py       # test the language parser
python openvino_benchmark.py       # run the Intel inference benchmark
python visual_grasp_debug.py       # render the scripted controller's attempt
```

## Team / track

Built for the Intel Physical AI Online Challenge, AI Infra Summit Hackathon
2026 (lablab.ai), by Muhammad.

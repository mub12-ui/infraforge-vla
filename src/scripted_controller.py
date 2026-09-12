"""
InfraForge VLA — Scripted Controller

Converts a list of ActionSteps (from language_interface.parse_command)
into an executable waypoint sequence for the bimanual MuJoCo environment,
and runs it, reporting reward/success at each step.

This is the piece that connects "language in" to "robot motion out" --
the seam other modules plug into. Swapping this rule-based waypoint
executor for a trained VLA policy later means replacing
`build_phases_from_steps()` and the execution loop, while
language_interface.py's output format (ActionStep list) stays the same.

KNOWN LIMITATION (documented honestly, see README): the pick-up phase
reliably reaches the target object but does not yet reliably retain
grip through the lift. Hand-off phases inherit this: they cannot
succeed if the initial pick-up didn't. This module still demonstrates
the full language -> plan -> execution -> reward pipeline end-to-end,
which is the architecture being evaluated, independent of final
grasp reliability.
"""

import numpy as np
from dm_control import mujoco as dm_mujoco
from gym_aloha.constants import ASSETS_DIR
from gym_aloha.tasks.sim_end_effector import TransferCubeEndEffectorTask

from language_interface import ActionStep, ActionType, ArmId, parse_command

# Weld offsets between mocap targets and actual gripper position,
# found by inspecting the MuJoCo model's equality-constraint data.
RIGHT_OFFSET = np.array([0.134706, 0.0, 0.00205])
LEFT_OFFSET = np.array([-0.134706, 0.0, 0.00205])
IDENT_QUAT = [1, 0, 0, 0]

# Fixed reference positions in this scene (single box, known start pose)
BOX_START = [0.2, 0.5, 0.05]
CENTER_HANDOFF = [0.0, 0.5, 0.20]


def _to_mocap(target, offset):
    return (np.array(target) - offset).tolist()


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def build_phases_from_steps(steps: list[ActionStep]):
    """
    Translate high-level ActionSteps into (name, n_steps, right_target,
    right_grip, left_target, left_grip) waypoints the executor understands.
    """
    phases = []
    # Track which side currently "holds" the object for hand-off phases
    for step in steps:
        if step.action == ActionType.PICK_UP:
            arm_is_right = step.arm == ArmId.A
            target = BOX_START
            other_target = [-0.18, 0.5, 0.295] if arm_is_right else [0.18, 0.5, 0.295]

            approach = [target[0], target[1], 0.15]
            descend = [target[0], target[1], 0.05]

            if arm_is_right:
                phases.append(("approach_pickup", 40, approach, 1.0, other_target, 1.0))
                phases.append(("descend_pickup", 40, descend, 1.0, other_target, 1.0))
                phases.append(("grasp", 50, descend, 0.0, other_target, 1.0))
                phases.append(("lift", 60, [target[0], target[1], 0.20], 0.0, other_target, 1.0))
            else:
                phases.append(("approach_pickup", 40, other_target, 1.0, approach, 1.0))
                phases.append(("descend_pickup", 40, other_target, 1.0, descend, 1.0))
                phases.append(("grasp", 50, other_target, 1.0, descend, 0.0))
                phases.append(("lift", 60, other_target, 1.0, [target[0], target[1], 0.20], 0.0))

        elif step.action == ActionType.HAND_OFF:
            # Move both arms to the center handoff point, transfer grip
            phases.append(("move_to_center", 50, CENTER_HANDOFF, 0.0, CENTER_HANDOFF, 1.0))
            phases.append(("transfer_grip", 30, CENTER_HANDOFF, 1.0, CENTER_HANDOFF, 0.0))

        elif step.action == ActionType.PLACE:
            phases.append(("place", 40, [0.0, 0.5, 0.05], 1.0, [-0.18, 0.5, 0.295], 1.0))

    return phases


def run_phases(physics, task, phases, verbose=True):
    """Execute a phase list, returning the reward trace."""
    right_mocap_cur = [0.31718881, 0.49999888, 0.29525084]
    left_mocap_cur = [-0.31718881, 0.5, 0.29525084]
    right_grip_cur = 1.0
    left_grip_cur = 1.0

    reward_trace = []
    for phase_name, n_steps, right_target, right_g, left_target, left_g in phases:
        right_mocap_target = _to_mocap(right_target, RIGHT_OFFSET)
        left_mocap_target = _to_mocap(left_target, LEFT_OFFSET)

        for t in range(n_steps):
            frac = (t + 1) / n_steps
            rp = _lerp(right_mocap_cur, right_mocap_target, frac)
            lp = _lerp(left_mocap_cur, left_mocap_target, frac)
            rg = right_grip_cur + (right_g - right_grip_cur) * frac
            lg = left_grip_cur + (left_g - left_grip_cur) * frac

            action = np.array(lp + IDENT_QUAT + [lg] + rp + IDENT_QUAT + [rg], dtype=np.float32)
            task.before_step(action, physics)
            physics.step(10)

        right_mocap_cur, left_mocap_cur = right_mocap_target, left_mocap_target
        right_grip_cur, left_grip_cur = right_g, left_g

        reward = task.get_reward(physics)
        reward_trace.append((phase_name, reward))
        if verbose:
            print(f"  [{phase_name}] reward={reward}")

    return reward_trace


def run_instruction(instruction: str, verbose=True):
    """Full pipeline: parse instruction -> build phases -> execute -> report."""
    if verbose:
        print(f"Instruction: {instruction}")

    steps = parse_command(instruction)
    if verbose:
        print("Parsed steps:")
        for s in steps:
            print(f"  {s}")

    xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
    physics = dm_mujoco.Physics.from_xml_path(str(xml_path))
    task = TransferCubeEndEffectorTask()
    task.initialize_episode(physics)

    box_start_idx = physics.model.name2id("red_box_joint", "joint")
    physics.data.qpos[box_start_idx:box_start_idx + 7] = BOX_START + [1, 0, 0, 0]
    physics.forward()

    phases = build_phases_from_steps(steps)
    if verbose:
        print(f"\nExecuting {len(phases)} phases...")

    reward_trace = run_phases(physics, task, phases, verbose=verbose)
    max_reward = max((r for _, r in reward_trace), default=0)

    if verbose:
        print(f"\nMax reward achieved: {max_reward} / 4")

    return {"steps": steps, "reward_trace": reward_trace, "max_reward": max_reward}


if __name__ == "__main__":
    run_instruction("pick up the mug with arm A, hand it to arm B")

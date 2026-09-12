"""
InfraForge VLA — Visual Grasp Debugger

Run this on YOUR machine (not in a headless environment) so you can
actually SEE where the gripper is relative to the box at each phase.
This is the same scripted attempt from our sandbox testing, but it
saves a frame after every phase so you can visually diagnose the
alignment problem instead of guessing blind.

Findings so far (from headless numeric testing):
  - Weld offset between mocap and gripper_link: right arm +0.1347 in x,
    +0.00205 in z (already applied below).
  - The gripper's BODY (gripper_bar / gripper_prop, not the fingers)
    is what's contacting the box -- meaning the approach height or
    orientation is off, not just position.

After running this, open the saved PNGs in order and look at:
  1. Is the gripper's fingers (not its palm/body) actually reaching
     the box?
  2. Are the fingers open WIDE ENOUGH to straddle the box before
     closing (box is 4cm wide total)?
  3. Is the approach height putting the finger PINCH POINT at the
     box's middle, not its top?
"""

import numpy as np
from dm_control import mujoco as dm_mujoco
from gym_aloha.constants import ASSETS_DIR
from gym_aloha.tasks.sim_end_effector import TransferCubeEndEffectorTask
from PIL import Image

xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
physics = dm_mujoco.Physics.from_xml_path(str(xml_path))
task = TransferCubeEndEffectorTask()
task.initialize_episode(physics)

box_start_idx = physics.model.name2id("red_box_joint", "joint")
physics.data.qpos[box_start_idx:box_start_idx + 7] = [0.2, 0.5, 0.05, 1, 0, 0, 0]
physics.forward()

IDENT_QUAT = [1, 0, 0, 0]
RIGHT_OFFSET = np.array([0.134706, 0.0, 0.00205])
LEFT_OFFSET = np.array([-0.134706, 0.0, 0.00205])


def to_mocap(target, offset):
    return (np.array(target) - offset).tolist()


def lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def save_frame(label):
    frame = physics.render(height=480, width=640, camera_id="angle")
    Image.fromarray(frame).save(f"grasp_debug_{label}.png")
    print(f"Saved grasp_debug_{label}.png")


right_mocap_cur = [0.31718881, 0.49999888, 0.29525084]
left_mocap_cur = [-0.31718881, 0.5, 0.29525084]
right_grip_cur = 1.0
left_grip_cur = 1.0

# TWEAK THESE if the images show the gripper is too high/low/off-center.
# z=0.035 was our best guess so far but the body-contact finding suggests
# it may need to be HIGHER (fingers pinching mid-box, not palm on top).
phases = [
    ("approach", 40, [0.2, 0.5, 0.15], 1.0, [-0.18, 0.5, 0.295], 1.0),
    ("descend",  40, [0.17, 0.5, 0.05], 1.0, [-0.18, 0.5, 0.295], 1.0),   # pulled x in from 0.2 -> 0.17 (was overshooting)
    ("grasp",    50, [0.17, 0.5, 0.05], 0.0, [-0.18, 0.5, 0.295], 1.0),
    ("lift",     60, [0.17, 0.5, 0.20], 0.0, [-0.18, 0.5, 0.295], 1.0),
]

save_frame("00_start")
for phase_name, n_steps, right_target, right_g, left_target, left_g in phases:
    right_mocap_target = to_mocap(right_target, RIGHT_OFFSET)
    left_mocap_target = to_mocap(left_target, LEFT_OFFSET)
    for t in range(n_steps):
        frac = (t + 1) / n_steps
        rp = lerp(right_mocap_cur, right_mocap_target, frac)
        lp = lerp(left_mocap_cur, left_mocap_target, frac)
        rg = right_grip_cur + (right_g - right_grip_cur) * frac
        lg = left_grip_cur + (left_g - left_grip_cur) * frac
        action = np.array(lp + IDENT_QUAT + [lg] + rp + IDENT_QUAT + [rg], dtype=np.float32)
        task.before_step(action, physics)
        physics.step(10)
    right_mocap_cur, left_mocap_cur = right_mocap_target, left_mocap_target
    right_grip_cur, left_grip_cur = right_g, left_g

    reward = task.get_reward(physics)
    print(f"[{phase_name}] reward={reward}")
    save_frame(phase_name)

print("\nOpen the grasp_debug_*.png files in order (00_start, approach, descend, grasp, lift)")
print("and look at where the fingers are relative to the box at 'descend' and 'grasp'.")

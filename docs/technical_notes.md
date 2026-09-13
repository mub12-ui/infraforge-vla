# Technical Notes: Grasp Geometry, Robustness, and the Hand-off Blocker

This document records the actual engineering findings behind
`scripted_controller.py` — the real measurements and bugs found, not a
theoretical description. Written for anyone continuing this work (including
future us).

## 1. Discovery: the gripper approaches sideways, not top-down

Our first assumption was that a top-down grasp (gripper descending onto an
object from above) was correct, since that's the intuitive mental model for
"picking something up." This was wrong.

By directly measuring the finger geometry (`geom_xpos` for
`10_left_gripper_finger` / `10_right_gripper_finger` relative to
`gripper_link`) at the identity quaternion `[1,0,0,0]`, we found the finger
centroid sits offset from `gripper_link` primarily along the **X axis**
(roughly -0.093 in x for the right arm), not below it in Z. This means at
identity orientation, the gripper's fingers point horizontally toward the
table's center, not downward. Every grasp attempt that assumed a downward
approach failed for this reason.

## 2. Discovery: a real 0.1347m offset in the mocap-to-gripper weld

The mocap point you command is not where the gripper physically ends up.
Inspecting the MuJoCo model's equality-constraint data directly
(`physics.model.eq_data`) revealed a weld constraint between
`mocap_right`/`mocap_left` and the corresponding `gripper_link`, with a
baked-in relative-pose offset:

```
right arm: eq_data ≈ [0, 0, 0, 0.1347, 0, 0.00205, ...]
left arm:  eq_data ≈ [0, 0, 0, -0.1347, 0, 0.00205, ...]
```

Commanding mocap directly to a desired gripper position (without
subtracting this offset) sends the gripper roughly 13cm away from intended.

## 3. Discovery: even with the correct math, an empirical residual remains

Combining the finger-offset and weld-offset math analytically gets you
close, but not exact — likely due to weld/actuator settling dynamics that
aren't captured by the static offset alone. We measured the real residual
error by commanding a naive computed target and directly comparing actual
vs. intended finger position, then corrected for that specific residual.
This two-step process (compute analytically, then measure and correct the
remainder) was more reliable than trying to get a fully analytical answer.

## 4. Discovery: approaching too close while open still causes a collision

Early attempts moved the gripper directly to the intended final position
while still open, then closed. This often knocked the object away before
closing began, because the *open* gripper's span already overlapped the
object's volume at that position. The fix: stage the approach — arrive at
a position offset further back (a "wide" position) with the gripper open,
settle, THEN move in to the final contact point, THEN close. Skipping the
settle step or moving too far in one motion reintroduces the collision.

## 5. Position-relative generalization

Once the above was solved for one calibration point (`BOX_START =
[0.2, 0.5, 0.05]`), we generalized it: rather than hardcoding the working
contact coordinates, we computed the fixed **delta** between the object's
position and the verified working contact target:

```
RIGHT_CONTACT_DELTA = [-0.145906, 0.0, -0.02215]
LEFT_CONTACT_DELTA  = [ 0.146,    0.0, -0.0222 ]
```

For any new object position, `contact_target = object_position + DELTA`.
This is what makes the pick-up robust to placement variation rather than
only working at one exact spot — see the robustness evaluation below.

## 6. Measured robustness boundaries (not assumed)

We empirically swept perturbations around the calibration point rather
than assuming a symmetric "safe radius." The actual tested boundary is
asymmetric:

| Direction | Safe range | Fails at |
|---|---|---|
| +x | up to +1.5cm | +2.0cm (reward drops to 1) |
| -x | up to -2.5cm (tested; may extend further) | not found within tested range |
| +y | up to +2.5cm (tested; may extend further) | not found within tested range |
| -y | up to ~-1.4cm | -2.0cm (reward drops to 0) |

The 10-seed evaluation (`robustness_eval.py`) randomizes within
`x ∈ [-2.5cm, +1.5cm]`, `y ∈ [-1.5cm, +2.5cm]` — the honestly-tested safe
window, not a guessed or symmetric range. Result: 9/10 seeds succeeded;
the one failure (seed 42, y offset -1.40cm) landed right at the known
edge of that window, which is corroborating evidence the boundary is real.

## 7. The hand-off blocker (unsolved, root-caused)

Attempting hand-off (right arm holds the object aloft, left arm approaches
to take it, right releases) surfaced a different problem than positioning:

- The right arm's grip proved genuinely **robust** — it held the object
  through the left arm's entire approach motion without dropping it, even
  when that approach physically disturbed the object's position.
- The left arm's approach was iteratively corrected using the same
  measure-then-correct method as above, getting from "missing entirely"
  down to roughly 1.2–1.5cm from contact.
- Critically: in every attempted release, commanding the right arm's grip
  from closed (0.0) back to open (1.0) did **not** result in the object
  separating from the right gripper. Contact-pair inspection after the
  commanded release consistently showed the object still only in contact
  with the right arm's fingers, never the left's.

This means the immediate blocker is not left-arm precision — it's that the
right arm's release mechanism isn't behaving as expected in this
held-in-air configuration. This is a well-defined, bounded next problem:
investigate why commanding `right_grip=1.0` doesn't produce actual finger
separation when the object is suspended (versus resting on the table,
where open/close was never an issue). Possible next steps: check whether
the gripper's actuator range/limits differ when off-table-support, or
whether the object is wedged such that the geometric opening isn't enough
to clear it at this specific orientation.

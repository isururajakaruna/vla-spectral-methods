# Smooth Operator

Python / PyBullet foundation for the CS5478 frequency-conditioned FAST policy project.

## Setup

The initial environment uses Python 3.9.6 (available on this Mac). Use a dedicated
virtual environment; model training will have its own environment once GPU access
is confirmed. PyBullet 3.2.7 builds from source on Apple Silicon and requires Xcode
Command Line Tools (`xcode-select --install` if missing).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel setuptools==75.8.0
.venv/bin/python -m pip install numpy==1.26.4 Pillow==11.3.0
ARCHFLAGS='-arch arm64' CFLAGS='-Dfdopen=fdopen' CXXFLAGS='-Dfdopen=fdopen' .venv/bin/python -m pip install --no-build-isolation pybullet==3.2.7
.venv/bin/python -m pip install --no-build-isolation -e .
```

On Linux, omit the three compiler environment variables. The source build can take
several minutes. The `fdopen` flag prevents bundled legacy zlib from replacing the
macOS SDK's standard `fdopen` declaration with a macro.

## Run

```sh
# Desktop simulation: approach A, attach, transport to B, release, retreat.
.venv/bin/python -m smooth_operator.demo

# Repeatable run without a window.
.venv/bin/python -m smooth_operator.demo --headless --seed 42

# Physics integration test.
.venv/bin/python -m unittest discover -s tests -v
```

The GUI runs once and stays open for ten seconds after placement; use `--hold 60`
to extend this. Camera images at the initial, attached, and placed stages and a
JSON result are saved under `artifacts/demo/`. Change this with `--output PATH`.
Every run chooses a fresh random seed unless `--seed INTEGER` is supplied. The
seed, selected cells, obstacle geometry, and obstacle contact count are saved in
the result JSON. Reuse that seed to reproduce the same scene.

## Robot and simulation

- Fixed base; two powered revolute joints rotate about Z, moving in the XY plane.
- Link lengths: 0.40 m and 0.32 m. Physics and motor commands: 240 Hz.
- A fixed magnetic tool adds no movement degree of freedom. A proximity-gated
  fixed constraint attaches a dynamic object; release removes the constraint.
  This abstracts grasping and vertical lifting; the object travels at table height.
- The table has a 17-column × 13-row workspace grid with 10 cm cells. Amber is
  the randomized pickup cell; green is the randomized goal cell. Cell indices are
  zero-based from the lower-left corner (-0.70, -0.65) m.
- Distinct pickup and goal centers are sampled at least 20 cm apart, within a
  20–65 cm radius, and checked for a safe route from the home pose. Grid highlights
  are visual only; the object rests on the flat table, not on raised platforms.
- A fixed purple obstacle at (0.35, 0) m measures 9 × 13 × 20 cm and physically
  blocks both the arm and the carried object.
- Analytical inverse kinematics, joint-space A*, and quintic joint interpolation
  drive the demo. The planner first tries a direct route, then a 0.1-radian grid;
  edges are checked at intervals no larger than 0.02 radians. Obstacle inflation
  accounts conservatively for link width, tool, payload, and tracking clearance.
  Planning is restricted to one elbow branch and bounded joint angles; sampling
  uses the connected region reachable from home, not every possible arm pose.
  Joint position motors execute the commands through physics; the moving robot
  and carried object are not teleported along the trajectory.
- A fixed overhead RGB camera supports both GUI and headless operation.
- After release, retreat, and one second of settling, placement coverage is
  `100 × area(object footprint ∩ goal cell) / area(object footprint)`.
  Error is `100 − coverage` percent; success requires release and at least 99%
  coverage. The demo additionally requires zero obstacle-contact steps, counting
  robot or object contact at each physics step. The calculation projects all eight box corners using the measured
  pose, then clips their convex hull to the goal cell, accounting for rotation
  and tilt. It uses simulator geometry rather than visible camera pixels, so
  arm occlusion does not affect the score. This is a planar area metric, not
  a 3D volume or grasp-quality metric.

The visible XY grid is workspace, not the arm's joint configuration space
(`theta1`, `theta2`). Some cells are unreachable; the demo uses two reachable
cells. The current joint-space planner uses known geometry; camera-derived
configuration-space mapping remains a later milestone.

This milestone is a scripted simulation check, not a learned VLA or the native
FAST experimental baseline. Perception, policy integration, conditioning, and
comparative metrics are subsequent work. The sampled planner and clearance margin
are appropriate for this static demo, not a continuous collision-safety guarantee.

`PlanarArm` exposes joint state, endpoint and object position, camera capture,
joint commands, stepping, attachment, and release for subsequent policy adapters.
Each instance owns its PyBullet client and should be used as a context manager.

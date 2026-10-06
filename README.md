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
.venv/bin/python -m smooth_operator.demo --headless

# Physics integration test.
.venv/bin/python -m unittest discover -s tests -v
```

The GUI runs once and stays open for ten seconds after placement; use `--hold 60`
to extend this. Camera images at the initial, attached, and placed stages and a
JSON result are saved under `artifacts/demo/`. Change this with `--output PATH`.

## Robot and simulation

- Fixed base; two powered revolute joints rotate about Z, moving in the XY plane.
- Link lengths: 0.40 m and 0.32 m. Physics and motor commands: 240 Hz.
- A fixed magnetic tool adds no movement degree of freedom. A proximity-gated
  fixed constraint attaches a dynamic object; release removes the constraint.
  This abstracts grasping and vertical lifting; the object travels at table height.
- Pickup A: (0.43, -0.25) m. Placement B: (0.43, 0.25) m.
- Analytical inverse kinematics and quintic joint interpolation drive the demo.
  Joint position motors execute the commands through physics; the moving robot
  and carried object are not teleported along the trajectory.
- A fixed overhead RGB camera supports both GUI and headless operation.
- Placement succeeds when the released object's XY position is within 3 cm of B
  after one second of settling.

This milestone is a scripted simulation check, not a learned VLA or the native
FAST experimental baseline. Perception, configuration-space planning, obstacles,
policy integration, conditioning, and comparative metrics are subsequent work.
The current motion script assumes an obstacle-free scene and does not provide
collision-aware planning.

`PlanarArm` exposes joint state, endpoint and object position, camera capture,
joint commands, stepping, attachment, and release for subsequent policy adapters.
Each instance owns its PyBullet client and should be used as a context manager.

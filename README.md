# Smooth Operator

Smooth Operator studies whether a learned Vision-Language-Action (VLA) policy can reproduce and later smooth collision-free robot motion. The simulator uses a three-joint SCARA arm (two planar revolute joints and one vertical prismatic wrist) to move an Apple, Coke can, or Pen through randomized grid obstacles. An unsmoothed, eight-connected A* trajectory is the expert and comparison baseline.

## Current flow and status

```text
Natural-language task + RGB image + robot state
                      |
       SmolVLA predicts 10-action chunks
                      |
     simulator validates every predicted action
              /                       \
       action is safe             action is unsafe
              |                       |
       execute SmolVLA        stop and fall back to A*
```

The Python 3.9 simulator generates collision-free A* demonstrations at 10 Hz. Each frame contains the workspace RGB image, instruction, joint state `[shoulder, elbow, wrist, gripper]`, and the corresponding absolute action. A Python 3.12 process converts these episodes to LeRobot format, trains SmolVLA, and serves inference to the simulator over a local socket.

Completed so far:

- Collision-aware low-height 2D manipulation, randomized objects and rectangular obstacles, A* planning, safety validation, metrics, and trajectory graphs.
- 600 valid demonstrations generated and converted: 468 training, 71 validation, and 61 held-out evaluation episodes.
- The SmolVLA download, conversion, training, and socket pipelines have passed smoke tests.

SmolVLA has **not yet been fully trained on the 600 episodes**. Only a one-step CPU training smoke test was completed, so the present learned checkpoint should not be used for a performance comparison.

## Setup

Create separate environments because the simulator uses Python 3.9 while LeRobot/SmolVLA requires Python 3.12. Use `python3.9` and `python3.12` where those commands are available; the Windows Python Launcher equivalents are `py -3.9` and `py -3.12`.

```bash
python3.9 -m venv .venv
python3.12 -m venv .venv312
```

Activate `.venv`, then install the simulator:

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

Activate `.venv312`, then install the training tools:

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e ".[vla]"
```

On macOS and Linux, activate an environment with `source .venv/bin/activate` or `source .venv312/bin/activate`. On Windows PowerShell, use `.venv\Scripts\Activate.ps1` or `.venv312\Scripts\Activate.ps1`. If PyBullet has no wheel for the platform, install the platform compiler toolchain before installing the simulation environment.

All commands below are the installed console commands and are independent of the operating system once the relevant environment is active. MPS is PyTorch's Apple GPU backend. `--device cpu` is portable but slow; `--device mps` requires compatible Apple hardware and a PyTorch installation where MPS is available.

## Run and produce graphs

Run the deterministic baseline and save its result plus a graph of X/Y position, planar speed, and the XY path around obstacles:

```bash
smooth-operator --headless --move apple --chart artifacts/astar_apple.png --result artifacts/astar_apple.json

# Replace apple with coke_can or pen. Omit --headless to watch the simulation.
smooth-operator --move pen --hold 10 --chart artifacts/astar_pen.png --result artifacts/astar_pen.json
```

Objects can also be moved using natural-language prompts. With `--controller astar`, the prompt resolver determines the object and destination, and A* executes the motion:

```bash
smooth-operator --prompt "Move the apple to the target" --controller astar --chart artifacts/prompt_apple.png --result artifacts/prompt_apple.json

smooth-operator --prompt "Place the pen to the right of the Coke can" --controller astar --result artifacts/prompt_pen.json
```

Supported instructions identify an Apple, Coke can, or Pen and either the target/goal/destination or a spatial relation such as left, right, in front of, or behind another object. SmolVLA will accept broader paraphrases after full fine-tuning.

Run tests and an A* evaluation over held-out prompts and scene seeds:

```bash
python -m unittest discover -s tests -v
smooth-operator-evaluate --controller astar --episodes 100 --output artifacts/astar_evaluation.json
```

The PNG files show the current unsmoothed baseline and expose abrupt changes at A* corners. The JSON files contain path length, planning time, velocity, acceleration, jerk, collision count, and success data. After full SmolVLA training, run the learned evaluation below and compare its JSON metrics with `astar_evaluation.json`. Learned trajectory plots are the next reporting task; currently the automatic PNG plot is produced by the A* controller.

## Data, training, and learned evaluation

Generate more expert episodes in Python 3.9, then convert them in Python 3.12:

```bash
smooth-operator-generate-raw --episodes 600 --output datasets/raw-600

smooth-operator-convert --input datasets/raw-600 --output datasets/lerobot-600 --repo-id local/smooth_operator
```

Fine-tune on the converted training split:

```bash
smooth-operator-train --dataset-repo local/smooth_operator_train --dataset-root datasets/lerobot-600/train --output checkpoints/smolvla --steps 20000 --device cpu
```

Use `--device mps` instead of `--device cpu` on a compatible Apple system. Model files are cached in the default Hugging Face cache unless `HF_HOME` is configured for the shell.

After training, start inference in the Python 3.12 terminal:

```bash
smooth-operator-inference-server --checkpoint checkpoints/smolvla/checkpoints/last/pretrained_model --device cpu
```

Then evaluate from a second, Python 3.9 terminal:

```bash
smooth-operator-evaluate --controller smolvla --episodes 100 --checkpoint checkpoints/smolvla/checkpoints/last/pretrained_model --output artifacts/smolvla_evaluation.json
```

For an individual learned prompt, keep the inference server running and execute this in the simulation environment:

```bash
smooth-operator --prompt "Carry the Coke can to the destination" --controller smolvla --checkpoint checkpoints/smolvla/checkpoints/last/pretrained_model --result artifacts/smolvla_prompt.json
```

## Next steps

1. **Train the 600-episode dataset on suitable compute.** The current machine reported MPS as unavailable; the measured CPU rate implies roughly 92 hours for 20,000 steps. A working Apple MPS setup would make iteration practical. The training wrapper must gain CUDA device support before using an NVIDIA GPU.
2. **Select and evaluate checkpoints.** Add validation-based checkpoint selection, then measure learned-only success, fallback rate, collision-free success, prompt success, latency, path length, and jerk on held-out seeds.
3. **Collect failure-focused data.** Add unsuccessful objects, prompts, destinations, and obstacle layouts to the dataset and resume training until learned-only completion reaches the target.
4. **Generate direct comparison plots.** Plot A*, learned-only, and fallback trajectories together, including velocity, acceleration, jerk, and aggregate success metrics.
5. **Implement frequency-domain conditioning.** Use the unsmoothed A* demonstrations as the baseline, suppress undesirable high-frequency action components, retrain or condition the policy, and compare smoothness without reducing task or collision-free success.

Datasets, model weights, checkpoints, and generated artifacts stay outside source control because they are large and machine-specific.

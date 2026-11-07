# Smooth Operator

Smooth Operator studies whether a learned Vision-Language-Action (VLA) policy can reproduce and later smooth collision-free robot motion. It includes the original three-joint SCARA experiment and an official SO-101 arm for fine-tuning `lerobot/smolvla_base`. Both move an Apple, Coke can, or Pen through grid obstacles; unsmoothed eight-connected A* is the comparison baseline.

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
       execute SmolVLA          stop and report failure
```

The Python 3.9 simulator generates collision-free A* demonstrations. SO-101 episodes contain three `256×256` RGB observations and calibrated six-dimensional states/actions at 30 Hz. A Python 3.12 process converts these episodes to LeRobot format, trains SmolVLA, and serves inference to the simulator over a local socket.

Completed so far:

- Collision-aware low-height 2D manipulation, randomized objects and rectangular obstacles, A* planning, safety validation, metrics, and trajectory graphs.
- 600 valid SCARA demonstrations generated and converted: 468 training, 71 validation, and 61 held-out evaluation episodes.
- The SmolVLA download, conversion, training, and socket pipelines have passed smoke tests.
- The SO-101 path supplies the model's native six-dimensional state/action interface plus overhead, wrist, and side cameras. It supports matched A* and learned-only runs without safety fallback hiding model failures.

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

Run the matched SO-101 A* baseline:

```bash
smooth-operator --robot so101 --controller astar --seed 42 --prompt "Move the apple to the target" --chart artifacts/so101_astar.png --result artifacts/so101_astar.json
```

The web studio (`smooth-operator-web`) requires HTTP Basic auth on every route; set `SMOOTH_OPERATOR_USER` and `SMOOTH_OPERATOR_PASSWORD` before starting it. It starts and stops the inference server itself. It looks for the Python 3.12 interpreter in `SMOOTH_OPERATOR_VLA_PYTHON`, then a sibling conda env named `smooth-operator-vla`, then `.venv312`. Fine-tuned checkpoints under `checkpoints/**/pretrained_model` appear in its model list automatically.

The studio's models are defined in `src/smooth_operator/vla_models.py`. Each model appears under the tab that matches its native action output. The **Time domain** tab holds the continuous-action models (SmolVLA, MolmoAct2 SO-100/101, π0.5 and π0), which run with the `direct` decoder. The **Frequency domain** tab holds the models that predict FAST tokens (DCT coefficients). Their DCT-family decoders (`fast_idct`, `band_idct`, `custom_hybrid`) invert the coefficients the model itself predicted. Models that cannot drive the SO-101 joints are listed but disabled, each with its reason. The π models need the gated `google/paligemma-3b-pt-224` tokenizer: accept its license on Hugging Face and run `hf auth login` in the Python 3.12 environment. They also need `pip install protobuf sentencepiece`.

Each model in the list is a family with one checkpoint per environment. Step 2, **Select Environment**, offers two kinds of environment. The LIBERO benchmark suites (Object, Spatial, Goal, Long) run a Franka Panda in MuJoCo, and LeRobot publishes LIBERO fine-tunes of SmolVLA, MolmoAct2, π0.5, π0, π0-FAST and X-VLA for them. The custom PyBullet setups are the SO-101 and SCARA arrangements. Selecting a model opens its default environment: a LIBERO suite when it has a LIBERO checkpoint, otherwise SO-101. You can then switch to any other environment. A ✓ marks environments where the model has a runnable checkpoint, and the matching checkpoint is loaded on the inference server. In a LIBERO suite, each prompt chip is one benchmark task and shows that task's initial scene. The episode runs closed-loop inside the inference server (`smooth_operator.libero_env`), and success is LIBERO's own goal check.

The studio also serves `/terminal`, an interactive shell at the project root with live output (xterm.js on a pseudo-terminal, shared by every open terminal page), and `/browse`, a file browser that downloads files, uploads files or whole folders, and creates folders. The file browser never deletes or overwrites anything, but the terminal can run any command as your user. Both only answer requests from the studio's own pages and are disabled when `--host` is not a loopback address unless you pass `--allow-remote-tools`. Anyone who can reach the port then gets a shell.

LIBERO goes in the Python 3.12 environment: `pip install --no-deps hf-libero==0.1.4 robomimic==0.2.0`, then their remaining requirements except `hf-egl-probe`/`egl_probe`, `wandb` and `tensorboard`, none of which are needed for evaluation: `pip install robosuite==1.4.0 mujoco bddl easydict einops future hydra-core thop h5py imageio imageio-ffmpeg tensorboardX termcolor`. The `egl_probe` packages need a C compiler to build. Then import `libero` once and answer its config prompt. Rendering uses `MUJOCO_GL=egl`.

StarVLA Qwen2.5-VL-FAST (`StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1`) is not a LeRobot policy. It runs through StarVLA's own code, which the inference server expects at `../third_party/starVLA` (next to this repository) or at `$STARVLA_ROOT`: `git clone https://github.com/starVLA/starVLA ../third_party/starVLA`. Its model code needs nothing beyond the Python 3.12 environment. The checkpoint (8 GB) and its base VLM `StarVLA/Qwen2.5-VL-3B-Instruct-Action` (7.5 GB) download on first use.

If the GPU driver is older than the CUDA build PyTorch ships with, reinstall PyTorch from the matching index, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

For SmolVLA, start the Python 3.12 inference server with the trained checkpoint:

```bash
smooth-operator-inference-server --checkpoint checkpoints/so101-smolvla/checkpoints/020000/pretrained_model --device auto
```

Then run the same scene and prompt from the Python 3.9 simulation environment:

```bash
smooth-operator --robot so101 --controller smolvla --learned-only --seed 42 --checkpoint checkpoints/so101-smolvla/checkpoints/020000/pretrained_model --prompt "Move the apple to the target" --chart artifacts/so101_smolvla.png --result artifacts/so101_smolvla.json
```

Run a matched multi-seed comparison while the server is running:

```bash
smooth-operator-compare --episodes 5 --checkpoint checkpoints/so101-smolvla/checkpoints/020000/pretrained_model --charts artifacts/so101-comparison --output artifacts/so101-comparison.json
```

This produces per-controller trajectory and joint graphs, raw learned actions, failures, aggregate success and collision rates, endpoint error, path length, jerk, and inference-versus-planning latency. SmolVLA is evaluated without A* fallback; a rejected action is recorded as a learned failure.

The original SCARA commands remain available:

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

Generate a pilot and full SO-101 expert dataset in Python 3.9, then convert it in Python 3.12:

```bash
smooth-operator-generate-raw --robot so101 --episodes 50 --output datasets/so101-pilot
smooth-operator-generate-raw --robot so101 --episodes 600 --output datasets/so101-raw-600

smooth-operator-convert --input datasets/so101-raw-600 --output datasets/so101-lerobot --repo-id local/smooth_operator_so101
```

Fine-tune on the converted training split:

```bash
smooth-operator-train --dataset-repo local/smooth_operator_so101_train --dataset-root datasets/so101-lerobot/train --output checkpoints/so101-smolvla --steps 20000 --device auto
```

`--device auto` selects CUDA, then MPS, then CPU. Model files are cached in the default Hugging Face cache unless `HF_HOME` is configured for the shell.

After training, start inference in the Python 3.12 terminal:

```bash
smooth-operator-inference-server --checkpoint checkpoints/so101-smolvla/checkpoints/020000/pretrained_model --device auto
```

Then run the matched SO-101 comparison from a second, Python 3.9 terminal:

```bash
smooth-operator-compare --episodes 5 --checkpoint checkpoints/so101-smolvla/checkpoints/020000/pretrained_model --charts artifacts/so101-comparison --output artifacts/so101-comparison.json
```

For an individual learned prompt, keep the inference server running and execute this in the simulation environment:

```bash
smooth-operator --robot so101 --prompt "Carry the Coke can to the destination" --controller smolvla --learned-only --checkpoint checkpoints/so101-smolvla/checkpoints/020000/pretrained_model --result artifacts/so101-smolvla-prompt.json
```

## Next steps

1. **Generate and train the SO-101 dataset on suitable compute.** The existing 600 episodes belong to SCARA and remain archived. Use CUDA for the full SO-101 run where available.
2. **Select and evaluate checkpoints.** Measure learned-only success, collision-free success, prompt success, latency, path length, and jerk on held-out seeds.
3. **Collect failure-focused data.** Add unsuccessful objects, prompts, destinations, and obstacle layouts to the dataset and resume training until learned-only completion reaches the target.
4. **Compare A* and learned-only SmolVLA.** Use identical prompts and seeds and retain the trajectory, joint, velocity, acceleration, jerk, endpoint, and aggregate success outputs.
5. **Implement frequency-domain conditioning.** Use the unsmoothed A* demonstrations as the baseline, suppress undesirable high-frequency action components, retrain or condition the policy, and compare smoothness without reducing task or collision-free success.

Datasets, model weights, checkpoints, and generated artifacts stay outside source control because they are large and machine-specific.

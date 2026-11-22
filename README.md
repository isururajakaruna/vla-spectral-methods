# Smooth Operator: Spectral Action Tokens for Vision-Language-Action Adaptation

> **Quick navigation for reviewers**
> - **Final Report (PDF):** [`report/report.pdf`](report/report.pdf) ([LaTeX source](report/report.tex))
> - **Interactive Experiment Dashboard (HTML):** [`experiment/reports/training_report.html`](experiment/reports/training_report.html) | [Experiment Trail](experiment/reports/experiment_trail.html)
> - **Core Spectral Codec & Policy Code:** [`experiment/adaptive_dwt_vla/src/lerobot_policy_adaptive_dwt/`](experiment/adaptive_dwt_vla/src/lerobot_policy_adaptive_dwt/)
> - **Measured LIBERO Results (400-ep JSONs & Summary):** [`experiment/adaptive_dwt_vla/results/summary.md`](experiment/adaptive_dwt_vla/results/summary.md)
> - **Released LoRA Adapter Checkpoints (Git LFS):** [`experiment/adaptive_dwt_vla/weights/`](experiment/adaptive_dwt_vla/weights/)
> - **Presentation Slides (HTML):** [`presentations/single/smooth_operator_deck.html`](presentations/single/smooth_operator_deck.html)

Smooth Operator investigates how spectral representations of robot action chunks affect end-to-end learning efficiency and token structure when adapting a pretrained Vision-Language-Action (VLA) policy (`lerobot/pi0fast-base`) with LoRA on the **LIBERO** manipulation benchmark.

Under a shared calibration rule, 302-token vocabulary, and finite-state grammar mask (zero malformed streams), we compare localized spectral bases (**Haar wavelets**, fixed and adaptive) against global smooth bases (**DCT-II without BPE** and **discrete orthogonal Gram polynomials**), along with scale-aware loss variants.

- **Synthetic & offline study reports:** [`wavelet_selection.html`](experiment/reports/wavelet_selection.html), [`transform_shift_analysis.html`](experiment/reports/transform_shift_analysis.html), [`transform_fingerprint_analysis.html`](experiment/reports/transform_fingerprint_analysis.html), [`transform_topk_shift_analysis.html`](experiment/reports/transform_topk_shift_analysis.html), [`transform_polynomial_variants.html`](experiment/reports/transform_polynomial_variants.html)
- **Detailed experiment README & commands:** [`experiment/adaptive_dwt_vla/README.md`](experiment/adaptive_dwt_vla/README.md)

## Key Results (LIBERO, Closed-Loop Simulation)

### Full 40-Task LIBERO (400 episodes: 10 initial states x 40 tasks, seed 0)

| Model | Epoch 1 (14.5k steps) | Epoch 2 (29k steps) | Epoch 2 LIBERO-10 |
|---|---:|---:|---:|
| Haar DWT fixed (soft labels + step-weighted, $p=1$) | 53.0% (212/400) | 69.0% (276/400) | 46% |
| Haar DWT fixed (plain CE) | 58.5% (234/400) | 79.2% (317/400) | 54% |
| DCT-II, no BPE (plain CE) | 77.0% (308/400) | 85.5% (342/400) | 66% |
| DCT-II, no BPE ($\sqrt{\text{step}}$-weighted, $p=0.5$) | 79.5% (318/400) | 88.2% (353/400) | 70% |
| **Gram orthogonal polynomials ($\sqrt{\text{step}}$-weighted, $p=0.5$)** | **81.2% (325/400)** | **90.2% (361/400)** | **81%** |

- At **epoch 1**, Gram polynomials reach 81.2% overall and **91% on LIBERO-Goal** (vs. 72% for matched $\sqrt{\text{step}}$-weighted DCT, paired exact McNemar $p = 3.1\times 10^{-4}$).
- At **epoch 2**, Gram polynomials reach **90.2%** overall, significantly beating plain DCT (85.5%, $p = 0.020$) and plain Haar (79.2%, $p = 8.0\times 10^{-6}$), with **81% on LIBERO-10** (+15 points over plain DCT, $p = 0.011$).
- Synthetic token-shift analysis (400 smooth trajectories, 9 geometric edits, equal-accuracy quantizer) predicts this exact ordering: small smooth edits change **22%** of tokens in Gram polynomials, **30%** in DCT-II, and **69%** in Haar.

### Single-Task LIBERO-Object #0 (60 episodes: 3 seeds x 20 initial states, 2,000 updates)

| Variant | Success @1,000 | Success @2,000 | Tokens/chunk | Latency (ms) |
|---|---:|---:|---:|---:|
| Matched FAST (DCT + BPE) | 31/60 (51.7%) | 17/60 (28.3%) | 29.6 | 567 |
| Adaptive Haar (plain CE) | 26/60 (43.3%) | 39/60 (65.0%) | 61.6 | 1,024 |
| DCT-II, no BPE (plain CE) | 29/60 (48.3%) | 43/60 (71.7%) | 64.1 | 1,035 |
| Fixed Haar (soft + step-weighted) | 37/60 (61.7%) | 52/60 (86.7%) | 69.1 | 1,185 |
| **Fixed Haar (plain CE)** | **47/60 (78.3%)** | **53/60 (88.3%)** | 69.1 | 1,134 |

## Repository Layout

```text
smooth-operator/
├── report/                              # Final NeurIPS-style report (report.pdf, report.tex, figures/)
├── experiment/
│   ├── adaptive_dwt_vla/
│   │   ├── src/lerobot_policy_adaptive_dwt/  # Spectral codecs (Haar, DCT, Gram poly), FSM grammar, LoRA policy plugin
│   │   ├── scripts/                     # Data prep, offline analyses, training, LIBERO eval, report builders
│   │   ├── results/                     # Per-episode LIBERO JSONs, offline CSVs/JSONs, summary.md
│   │   ├── weights/                     # Released LoRA adapter checkpoints (Git LFS)
│   │   └── tests/                       # CPU and GPU unit tests
│   └── reports/                         # Self-contained HTML reports (training, trail, synthetic studies)
├── presentations/                       # Interactive HTML presentation slides and media assets
├── src/smooth_operator/                 # Web studio, LIBERO inference server, and legacy SCARA/SO-101 simulator
└── tests/                               # Simulator and studio unit tests
```

## Setup and Reproduction

### 1. Environment (Python 3.12 + CUDA GPU)

Tested on an NVIDIA RTX A6000 (48 GB) with Python 3.12, PyTorch 2.11 (`cu128`), `lerobot==0.6.1`, `transformers==5.5.4`, and `PyWavelets==1.8.0`.

```bash
conda create -n smooth-operator-vla python=3.12 -y
conda activate smooth-operator-vla

# Install base package and VLA dependencies
pip install -e ".[vla]"
pip install pywavelets pytest matplotlib protobuf sentencepiece

# Install LIBERO simulator dependencies (EGL headless rendering)
pip install --no-deps hf-libero==0.1.4 robomimic==0.2.0
pip install robosuite==1.4.0 mujoco bddl easydict einops future hydra-core thop h5py imageio imageio-ffmpeg tensorboardX termcolor

# Register the spectral-token policy plugin
cd experiment/adaptive_dwt_vla
pip install --no-deps --no-build-isolation -e .
```

The $\pi_0$-FAST base model requires access to the gated `google/paligemma-3b-pt-224` tokenizer on Hugging Face (`hf auth login`). Adapter checkpoints in `experiment/adaptive_dwt_vla/weights/` are stored with Git LFS (`git lfs pull`).

### 2. Run Unit Tests and Offline Analyses (CPU)

```bash
cd experiment/adaptive_dwt_vla

# Codec, grammar FSM, and oracle tests
pytest -q -m "not gpu"

# Prepare LIBERO index and per-task 90/10 split (seed 0)
python scripts/prepare_libero.py

# Offline wavelet ranking and synthetic token-shift / polynomial studies
python scripts/wavelet_selection.py
python scripts/transform_shift_analysis.py
python scripts/transform_fingerprint_analysis.py
python scripts/transform_topk_shift_analysis.py
python scripts/transform_polynomial_variants.py
```

### 3. Train and Evaluate on Full LIBERO (GPU)

```bash
cd experiment/adaptive_dwt_vla

# Train Gram orthogonal polynomials with sqrt-step weighting (epoch 1 = 14,500 steps)
python scripts/train.py --variant poly_fixed --weight-power 0.5 \
  --name full_s0_poly_fixed_sqrtw --seed 0 --tasks $(seq 0 39) \
  --horizon 10 --steps 29000 --stop-at 14500 \
  --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300

# Resume for epoch 2 (to 29,000 steps under the 43,500-step cosine)
python scripts/train.py --variant poly_fixed --weight-power 0.5 \
  --name full_s0_poly_fixed_sqrtw --seed 0 --tasks $(seq 0 39) \
  --horizon 10 --steps 43500 --stop-at 29000 --resume \
  --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300

# Closed-loop LIBERO evaluation (10 episodes/task across each suite)
for suite in libero_spatial libero_object libero_goal libero_10; do
  MUJOCO_GL=egl python scripts/evaluate_policy.py \
    --models ckpt:full_s0_poly_fixed_sqrtw/ckpt/step_29000 \
    --suite "$suite" --task-ids 0 1 2 3 4 5 6 7 8 9 \
    --episodes 10 --first-init-state 0 --videos 0 --label-prefix full40_
done

# Regenerate all summary tables and HTML reports
python scripts/write_reports.py
python scripts/write_training_report.py
python scripts/write_experiment_trail.py
```

Replace `--variant poly_fixed --weight-power 0.5` with `dct_fixed`, `dwt_fixed`, or `dwt_fixed_slw` to reproduce the DCT and Haar baselines. See [`experiment/adaptive_dwt_vla/README.md`](experiment/adaptive_dwt_vla/README.md) for single-task commands and queue scripts.

### 4. Recompile the Final Report

```bash
cd report
pdflatex report.tex && pdflatex report.tex
```

## Web Studio and Legacy Simulator

This repository also includes the interactive web studio (`smooth-operator-web`) and the earlier PyBullet SCARA/SO-101 A* and SmolVLA pipelines under `src/smooth_operator/`.

- **Web Studio:** set `SMOOTH_OPERATOR_USER` and `SMOOTH_OPERATOR_PASSWORD` (HTTP Basic auth is required on every route) and launch `smooth-operator-web`. It runs LIBERO and SO-101 closed-loop inference across time-domain and frequency-domain VLA checkpoints.
- **PyBullet SCARA / SO-101 baseline:** install `pip install -e .` in a Python 3.9 environment and run:
  ```bash
  smooth-operator --robot so101 --controller astar --seed 42 \
    --prompt "Move the apple to the target" \
    --chart artifacts/so101_astar.png --result artifacts/so101_astar.json
  ```

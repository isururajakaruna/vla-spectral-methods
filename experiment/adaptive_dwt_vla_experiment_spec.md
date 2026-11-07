# Implementation Brief: Adaptive-DWT Action Tokenization for a π₀-FAST-Style VLA

**Status:** Coding-agent implementation specification (research prototype)  
**Created:** 2026-11-06  
**Primary GPU:** 1 × NVIDIA A100 80 GB; **fallback:** 1 × RTX A6000 48 GB  
**Framework:** Hugging Face LeRobot (PyTorch) + LIBERO  
**Primary pretrained checkpoint:** `lerobot/pi0fast-base`  
**Reference baselines:** `lerobot/pi0fast-libero`, `lerobot/pi0_libero` (evaluate unmodified, with their documented configurations)

> **To the coding agent:** Implement and run the earliest feasible end-to-end experiment, not just scaffolding. Prioritize verified execution and meaningful measurements. Work autonomously where reasonable. Inspect the checked-out LeRobot version before using a method or CLI flag: APIs change. Keep all original LeRobot policy classes and pretrained checkpoints untouched. If an assumption below conflicts with the installed version, adapt it, document the change, and keep the experimental intent intact. Never report synthetic results as measured.

## 0. Objective and hypothesis

Research hypothesis: **An autoregressive vision-language-action policy can predict an action sequence more efficiently by emitting two control tokens before each temporal segment—one choosing the segment duration, the other choosing its discrete-wavelet-transform (DWT) resolution—followed by a variable number of quantized wavelet-coefficient tokens.** The generated token stream should reconstruct physically plausible actions and, ideally, retain LIBERO task success while reducing action-generation token count or latency.

**What we want to learn, in order:**

1. Can an adaptive DWT codec reconstruct LIBERO demonstration actions with acceptable position, velocity, and gripper-event fidelity?
2. Can the pretrained π₀-FAST autoregressive model be fine-tuned to predict this *new* sequence grammar from images, proprioception, and instruction?
3. Does the learned controller actually select useful segment durations/resolutions without seeing future ground truth?
4. Does the resulting policy complete LIBERO tasks, compared with a matched FAST baseline and a fixed-resolution DWT ablation?
5. Does it improve the **success / generated-token / inference-latency** trade-off?

Do **not** change the original π₀ flow-matching expert in Phase 1. π₀-FAST generates action tokens autoregressively through the Gemma language backbone; it does **not** use π₀'s separately trained ~300M flow-matching expert as its action decoder.

## 1. Non-negotiable research controls

- LeRobot is the implementation/evaluation framework; use OpenPI as an architecture/reference source only.
- Create a separate `adaptive_dwt` policy. Do **not** overwrite `pi0_fast`, `pi0`, their processors, or pretrained checkpoint artifacts.
- Start from `lerobot/pi0fast-base` for **both** FAST fine-tuning and the new DWT variants. Evaluate `lerobot/pi0fast-libero` and `lerobot/pi0_libero` as additional **pretrained reference** baselines. Clearly label reference vs compute-matched comparisons.
- For a fair tokenizer comparison: same training demonstration IDs, action parameterization, images, state, instructions, control frequency, normalization, action horizon, augmentation, optimizer schedule, effective batch, train steps or action-token exposure, and evaluation episodes. Log differences that cannot be eliminated.
- Critical: report **both** equal optimizer-step and equal GPU-hour comparisons, because sequence length changes computational cost. Record total seen demonstration chunks and total predicted action tokens.
- Use episode-level train/validation splits; never split temporally adjacent windows from the same episode across train and validation.
- Use only observations available at inference to condition the model. Offline “oracle” choices are allowed **as targets** during imitation training, but are never provided to the policy during rollout.
- Avoid claims of novelty or superiority until supported by actual results and a literature check.

## 2. Upstream repositories and documentation

Inspect these exact sources before editing:

- LeRobot repository: https://github.com/huggingface/lerobot
- LeRobot π₀-FAST implementation: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py
- π₀-FAST config: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/configuration_pi0_fast.py
- π₀-FAST preprocessing: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/processor_pi0_fast.py
- LeRobot π₀-FAST usage: https://huggingface.co/docs/lerobot/pi0fast
- Custom policy/plugin guide: https://huggingface.co/docs/lerobot/main/en/bring_your_own_policies
- LIBERO integration: https://huggingface.co/docs/lerobot/main/en/libero
- Install: https://huggingface.co/docs/lerobot/main/en/installation
- OpenPI official reference: https://github.com/Physical-Intelligence/openpi
- LIBERO dataset: https://huggingface.co/datasets/lerobot/libero

**Version policy:** clone LeRobot, record `git rev-parse HEAD`, pin the exact dependency lock/version set, record dataset revision and checkpoint revision/hash. Documentation sometimes has different examples on the `main` and stable branches. Verify the installed version's commands with `--help` and source inspection. Do not blindly mix commands from separate releases.

## 3. Environment: one A100 80GB, with A6000 fallback

Expected machine: Linux x86-64, CUDA-compatible driver, sufficient SSD space for model/dataset/checkpoints, optional headless EGL for LIBERO simulation.

First run:

```bash
nvidia-smi
python --version
which nvcc || true
mkdir -p runs data_cache results
export MUJOCO_GL=egl
```

Install a documented version of LeRobot from source using the current compatible Python/PyTorch/CUDA combination; install its `pi`, `training`, `libero`, evaluation extras, plus `pywavelets`, `pytest`, plotting utilities and optionally `peft` and `wandb` / TensorBoard. **Do not assume the latest PyTorch wheel works with the host's driver.** Make a smoke test that instantiates π₀-FAST, loads a LIBERO batch, runs one forward and one backward pass, and runs one LIBERO environment step.

Proposed operating mode:

- GPU: A100 80GB, BF16, initial microbatch 1–4, gradient accumulation to effective batch 16–32 if feasible.
- A6000 48GB fallback: BF16 if supported and stable, microbatch 1, gradient checkpointing, accumulation, selective freezing/LoRA. Reduce horizon or image resolution **only as explicitly recorded ablations**; preserve a matched FAST baseline.
- Prefer LoRA adaptation of relevant Gemma attention/MLP projections plus trainable new action-token embeddings/output parameters. Do not train all PaliGemma weights in the first run.
- Start with LoRA rank 8 or 16; tune only after a memory/throughput profile. Freeze vision encoder for the first pilot if needed. Verify actual parameter `requires_grad` flags and gradients.
- Measure real **seconds per optimizer step** and peak allocated/reserved GPU memory over at least 100 warmed-up steps. Do not hardcode previous estimates as expected benchmarks.

## 4. Experiment design and phases

| Experiment | Action representation | Purpose | Training |
|---|---|---|---|
| `ref_pi0` | Continuous flow-matching π₀ | Published-style reference | Evaluate pretrained LIBERO checkpoint, no modification |
| `ref_pi0fast` | Original DCT + FAST tokens | Published-style reference | Evaluate pretrained LIBERO checkpoint, no modification |
| `fast_matched` | Original FAST tokens | **Compute/data-matched baseline** | Fine-tune `pi0fast-base` on selected data |
| `dwt_fixed` | Fixed segments + fixed DWT detail budget | Is DWT a viable action tokenizer? | Same base and fine-tune recipe |
| `dwt_adaptive` | **Two learned control tokens per segment + DWT coefficients** | Test dynamic allocation of temporal detail | Same base and fine-tune recipe |

**Recommended staging:**

- **Stage A, offline codec:** no GPU training. Validate codec and compare against FAST on exactly the same normalized action chunks.
- **Stage B, overfit test:** one LIBERO task, very small fixed training set (e.g. 32–128 windows), prove near-perfect train-token prediction and valid action decoding. This catches architecture/label bugs faster than large training.
- **Stage C, first task pilot:** choose 1–2 LIBERO tasks with train/validation split by episode; train `fast_matched`, `dwt_fixed`, `dwt_adaptive` for 2k/5k/10k optimizer steps with checkpoints.
- **Stage D, closed-loop rollout:** evaluate matched checkpoints on exactly the same fixed initial states and seeds. Log videos and trajectories.
- **Stage E, benchmark expansion:** 10-step matched horizon first for direct π₀-FAST comparability, followed by **32-step** horizon for testing the hypothesized advantage of multiresolution. Retrain/reconfigure ALL compared methods at 32 steps. Do not compare a 32-step DWT model directly against a 10-step FAST model as if horizon were controlled.

A lower token count or lower reconstruction loss is not by itself evidence of good robotic control.

## 5. Adaptive DWT codec: precise MVP format

### 5.1 Inputs and normalization

LIBERO action is typically a 7D control vector: 6 continuous end-effector components and 1 gripper component. Confirm exact action dimensions/order, gripper convention, normalized ranges, relative/absolute action mode, control frequency, and whether the LeRobot checkpoint applies relative-action preprocessing. **Do not assume transforms or state representations align across checkpoints.**

Run **the same dataset/policy normalization** before codec comparison and the inverse transform after action decoding. Persist normalization statistics from the training split only. No sample-specific future normalization (would leak ground truth).

Apply DWT to the **6 continuous dimensions**. Preserve the gripper as an exact discrete event stream so wavelet approximation cannot silently erase grasp/release transitions. Use a separate small gripper grammar described below.

### 5.2 Chunking

Primary research action horizon `H=32`; fairness/compatibility horizon `H=10` as a separate experiment.

For H=32, allow segment lengths `L ∈ {4, 8, 16, 32}`, summing exactly to H. For H=10 use candidate lengths `{2, 4, 8}` or another explicit exact-cover scheme; **never pad future actions from the next episode**. Do not allow segments to cross episode boundaries. Keep the segment lengths powers of two so the Haar DWT is unambiguous.

### 5.3 The user's two-token controller

Before each segment emit exactly **two control tokens**:

1. `<LEN_4>` / `<LEN_8>` / `<LEN_16>` / `<LEN_32>`: how many action steps the following segment covers.
2. `<RES_0>` ... `<RES_J>`: how many successive wavelet detail levels to retain (where `J=log2(L)`).

These are **two positions in the output stream per segment**, not merely two token IDs in the entire vocabulary. A richer vocabulary represents the possible values of each position. A trained model predicts both positions autoregressively.

First implementation: Haar DWT (`pywt`, `mode='periodization'`), independent per continuous action dimension and per segment. Let `J=log2(L)` and write the coefficient order as `[cA_J, cD_J, cD_(J-1), ..., cD_1]`. Always keep `cA_J`; for resolution `k`, keep the `k` **coarsest** detail groups `cD_J ... cD_(J-k+1)`; omitted finer groups are zero. Thus the number of coefficients per continuous dimension is `2^k` for `k=0..J`. The inverse DWT reconstructs exactly L time steps. This is a well-defined progressive-resolution codec; localized adaptivity initially comes from **varying segment length and resolution across time**, not from independently selecting arbitrary wavelet-tree nodes.

### 5.4 Coefficient quantization and vocabulary

- Fit **per-dimension, per-coefficient-type** scale/clip calibrations using only training split. Avoid data-dependent per-window min/max scaling because the decoder will not know the future range.
- MVP: 256 scalar quantization bins (or fewer if vocabulary constraints require it), with deterministically specified round-to-nearest and saturation/clipping; record clipping rate. Test alternatives only after baseline works.
- Coefficient token type may use a shared 256-value vocabulary with a deterministic coefficient ordering, but calibrations must be reversible based on `(dimension, wavelet level, position)` known from the grammar. If that is not robust, use a typed coefficient vocabulary.
- The model's actual token IDs, embedding rows, and output logits must be handled explicitly. **Do not assume simply calling `tokenizer.add_special_tokens()` is sufficient** in LeRobot's custom PaliGemma/Gemma implementation. Inspect tied/untied embeddings and language head, create/initialize trainable representations of new IDs, and verify serialization/reload. An alternative is to map onto a documented unused/reserved subset of valid Gemma IDs, only after verifying absence of collisions with text, BOS/EOS, FAST and padding. Never hijack active text token IDs silently.
- Start with simple scalar tokens; optional learned BPE over wavelet tokens is a **later separate ablation**. FAST's original BPE compression is a strong baseline—explicitly report this difference rather than silently claiming equivalent compression machinery.

### 5.5 Gripper encoding

The gripper is encoded separately **inside each segment** with an unambiguous exact decoder:

- `<G0>` or `<G1>` for initial discretized gripper state.
- Zero or more `<GT_i>` tokens at strictly increasing offsets `i=1..L-1` indicating gripper state flips. (If the gripper has >2 valid states, redesign this deterministically.)
- `<G_END>` to terminate that segment's gripper events.

Quantize gripper using a fixed threshold derived from training metadata, and reconstruct the piecewise-constant binary sequence. Validate that all contact/gripper transitions match exactly for **oracle** encoded/decoded data.

### 5.6 Complete action grammar and decoding

An example segment grammar:

```text
<A_START>
  <LEN_8> <RES_2>
  [continuous wavelet coefficient tokens in deterministic dimension/scale order]
  <G0> [<GT_3> ...] <G_END>
  <LEN_16> <RES_1>
  [coefficients]
  <G1> [...] <G_END>
  ... (until accumulated segment lengths == action horizon)
<A_END>
```

Document and enforce whether each coefficient/token includes a position or has implicit position. Given `LEN`, `RES`, and dimensionality, the **exact number of subsequent coefficient tokens is known**. For all generated tokens use a finite-state grammar to constrain possible next-token IDs:

- At segment start, only legal `<LEN_*>` that fit the remaining horizon.
- Next, only legal `<RES_*>` for that length.
- Then exactly `6 * 2^k` coefficient tokens in fixed order.
- Then the gripper event sequence and required `<G_END>`.
- Allow `<A_END>` **only when exactly H action steps have been reconstructed**.
- Stop at `<A_END>`; reject/explain malformed or truncated streams. On invalid prediction, fallback to a safe explicit control policy (e.g., zero delta actions and no gripper transition) for the simulator, and record the event. Do not silently substitute ground-truth actions.
- Respect `max_action_tokens` with a checked worst-case bound and measured sequence-length histogram. No silent truncation.

Explicit BOS / prompt / target shifts must be unit tested: in next-token cross-entropy, the network position immediately before the first action-control token **must be supervised to predict it**. Verify there is no off-by-one where the first action token receives no loss.

### 5.7 Adaptive oracle teacher targets (for training only)

Construct the offline target sequence by optimizing over all legal segmentations and resolution choices using a **dynamic program** (rather than a greedy heuristic). Each candidate `(start, L, k)` has:

- Normalized continuous-action reconstruction loss in that segment (`MSE`, max absolute error and first-difference/velocity error).
- Optional acceleration/jerk penalty, but avoid excessive smoothing that removes real contact dynamics.
- Exact gripper-transition reproduction (hard constraint).
- Token cost including **both control tokens**, continuous coefficient tokens, gripper tokens and end marker.
- Optional contact-sensitive weighting, but infer contacts only from **training data annotations/trajectories** for oracle label construction, not as a hidden runtime input.

Example target objective (adjust weights on training split):

```
min_segmentation sum(
    alpha * weighted_reconstruction_error
  + beta  * velocity_error
  + gamma * number_of_serialized_tokens
)
subject to: segments cover H exactly; per-axis max error <= threshold;
            gripper-event fidelity is exact
```

Report the oracle Pareto frontier as error tolerance/token penalty changes. Choose hyperparameters using training + held-out validation, **not evaluation episodes**. Store oracle labels offline as training targets. At rollout, model must generate the two control tokens itself; **never use oracle decisions from the future trajectory**.

For `dwt_fixed`, choose a single fixed length/resolution on training data so that distortion is similar to the adaptive codec; this controls for the wavelet basis and tokenization quality.

## 6. Model adaptation in LeRobot

Implement an **out-of-tree LeRobot policy plugin** where supported (`lerobot_policy_adaptive_dwt`) or a minimal local policy registration if the pinned version does not support plugins. Subclass/copy only what is needed from `PI0FastPolicy` and its core `PI0FastPytorch`, avoiding upstream modifications. The key components are:

1. **Processor**: image/state/instruction preprocessing identical to π₀-FAST, action normalization, adaptive-DWT encoding into token IDs on train examples, and inverse decoding/postprocessing at inference.
2. **Policy model**: same pretrained PaliGemma backbone, autoregressive action token input/output, causal action attention, teacher-forced cross-entropy, and KV-cache generation if practical. Replace the FAST action tokenizer / action-token decoding and all FAST-specific termination assumptions.
3. **Trainable action token parameters**: new vocabulary rows / dedicated action embeddings and output projections plus LoRA as needed; log count of trained parameters. Verify learning of `<LEN_*>`, `<RES_*>`, coefficient bins and gripper grammar separately.
4. **Generation**: use a grammar-constrained token sampler and EOS as described. Prefer KV caching; evaluate without it if needed initially, but record inference impact. Do not call the old FAST detokenizer on wavelet tokens.
5. **Serialization**: checkpoints must include policy config, codec/quantization parameters, DWT wavelet/mode, all new token mappings, normalization stats, PEFT weights/modules-to-save, and any decoder state. Standalone reload must reproduce the same action from a saved test observation.

**Special caution:** in π₀-FAST, the Gemma LM predicts the action tokens; freezing the whole language model and updating no relevant logits/embeddings is insufficient. There is no independent 300M action expert to train. Do not reuse cached layer-KV tensors while their producing weights are being modified by LoRA. Frozen fixed visual embeddings are cacheable only if augmentation and preprocessing are also consistent.

### Suggested repository layout

```text
adaptive-dwt-vla/
├── README.md
├── pyproject.toml
├── configs/
│   ├── offline_codec.yaml
│   ├── fast_matched.yaml
│   ├── dwt_fixed.yaml
│   ├── dwt_adaptive.yaml
│   └── eval_libero.yaml
├── src/lerobot_policy_adaptive_dwt/
│   ├── __init__.py
│   ├── configuration_adaptive_dwt.py
│   ├── modeling_adaptive_dwt.py
│   ├── processor_adaptive_dwt.py
│   ├── codec.py
│   ├── quantizer.py
│   ├── oracle_dp.py
│   ├── token_grammar.py
│   └── metrics.py
├── scripts/
│   ├── inspect_environment.py
│   ├── inspect_lerobot_api.py
│   ├── prepare_libero.py
│   ├── evaluate_codec.py
│   ├── profile_training.py
│   ├── train.py
│   ├── evaluate_policy.py
│   └── summarize_results.py
├── tests/
│   ├── test_codec.py
│   ├── test_token_grammar.py
│   ├── test_oracle_dp.py
│   ├── test_model_loss.py
│   ├── test_checkpoint_reload.py
│   └── test_libero_smoke.py
├── data_cache/                  # gitignored
├── runs/                        # gitignored
└── results/
```

CLI entry points must actually work in the pinned environment. If LeRobot's `lerobot-train` CLI cannot load a custom plugin or add LoRA modules correctly, implement a small equivalent train entry point reusing LeRobot datasets, policy processors and evaluation logic. **Do not claim native CLI support without testing it.**

## 7. Data loading and evaluation

Use `lerobot/libero` and the existing LeRobot LIBERO environment integration. Keep **training and evaluation observation renaming** consistent; LeRobot documentation gives mappings such as:

```bash
--rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
```

Check actual feature names in the pinned dataset/checkpoint before adopting this map. The reference documented π₀-FAST setup uses `chunk_size=10`, `n_action_steps=10` for LIBERO; use these for the first fair comparison. Then run a separate 32-step experiment using **consistent execution horizon / replanning policy** across all compared models. For H=32 it is acceptable to execute 10 steps before replanning if done identically for all variants and explicitly documented.

Suggested first environment smoke run for **existing** pretrained FAST (check current CLI help and checkpoint conventions):

```bash
lerobot-eval \
  --policy.path=lerobot/pi0fast-libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
```

**Training data:** start with a clearly enumerated one-task subset. Confirm LIBERO demonstrations actually belong to the chosen task and don't accidentally pull 40-task data for one-task claims. Episode IDs, lengths, camera shapes, action conventions, and task instruction strings must be logged. Respect episode boundaries for action chunks. When padding at episode end is necessary, pad using an explicit mask; exclude padded steps from codec distortion and action loss.

**Primary closed-loop metric:** successes / episodes in the simulator using identical initial states/seeds. For an early smoke test, 5–10 episodes/task is sufficient for debugging, **not** for convincing comparisons. For pilot inference, prefer >=20 paired episodes/task across at least 2 tasks if compute allows. For a publishable comparison, use the standard four 10-task LIBERO suites, 10+ episodes/task, multiple seeds, and confidence intervals; distinguish training-seed variance from rollout-seed variance.

Also record videos and per-step telemetry: segment length, selected resolution, token count, parse errors, gripper transitions, action norms and trajectory smoothness.

## 8. Metrics to implement and persist

### Offline codec (oracle, zero model learning)

- Continuous-action normalized MSE / MAE, max absolute error, velocity MSE, acceleration/jerk error.
- Gripper-transition precision/recall and exact reconstruction rate.
- Serialized tokens per H-step chunk: **all** length, resolution, coefficient, gripper, BOS/EOS tokens. Count original FAST tokens under exactly the same chunk settings.
- Token ratio vs FAST, bits per action dimension/time step, total serialized bytes. If custom token vocabulary differs, state explicitly that token counts alone are not equal in information capacity; report entropy or estimated coding bits where feasible.
- Pareto plot: distortion vs average token count. Include fixed DWT, adaptive oracle DWT, and FAST.
- Boundary/contact error breakdown, quantizer clipping frequency, coefficient distribution by wavelet level.

### Learned model

- Training/validation next-token cross entropy + per-token-class losses/accuracies (`LEN`, `RES`, coeff, gripper, EOS).
- Invalid grammar output rate and percentage of chunks decoding to exactly H actions.
- Agreement of predicted vs oracle resolution/segment policies and **actual predicted-stream** reconstruction error. Oracle reconstruction measures only an upper bound for the codec.
- LIBERO task success, successes/episodes, 95% Wilson confidence interval, videos, mean return, per task/suite.
- Mean and p95 inference time for **prefill + all autoregressive decoding + inverse DWT**, GPU peak VRAM, measured train step wall-time, throughput and total GPU-hours.
- The crucial frontier: **task success vs generated action tokens vs inference latency vs GPU training budget**.

Report training steps, number of examples, random seeds and checkpoint names. Mark all outputs either measured, simulated/unit-test, or estimated.

## 9. Tests and acceptance gates

### Gate 1: codec correctness (must pass before training)

- `decode(encode(actions))` returns exact shape `[H, 7]`, correct units/normalization and valid bounds.
- Full resolution (`k=log2 L`) plus **lossless/no quantization test mode** reconstructs continuous actions to numerical tolerance; quantized mode reports measured (not asserted) error.
- Oracle gripper transitions are reproduced exactly.
- Explicit tests for H=10 and H=32, every allowed segment length/resolution, all-zero motion, constant motion, high-frequency motion, abrupt gripper changes, end-of-episode handling, and out-of-range coefficients.
- Parser rejects wrong coefficient counts, bad token IDs, illegal LEN/RES combos, gripper offsets outside segment, overrun/underrun, premature EOS, and missing EOS.
- DP oracle gives an equal-or-better objective than exhaustive enumeration on tiny synthetic cases.
- No future-derived labels or stats flow into inference inputs.

### Gate 2: model integration

- One batch forward pass yields finite loss; backward produces nonzero gradients for the intended trainable parameters.
- Unit test that teacher forcing supervises prediction of the **first** LEN token from the image/language/BOS prefix, not the next action token one position too late.
- Tiny-set overfit reaches very high token accuracy on train examples and can decode its own generated tokens.
- Save/reload model + tokenizer + normalizer + quantizer reproduces identical deterministic actions.
- Record peak GPU memory and measured iterations/sec on the actual machine.

### Gate 3: end-to-end robot success

- π₀-FAST reference runs in LIBERO before custom policy evaluation, verifying environment and action preprocessing.
- Custom model returns a valid action chunk; simulator accepts every action; at least one full episode runs without a parser crash.
- Track actual closed-loop successes. **Do not use declining training loss as proof of robot success.**
- Matched FAST / fixed-DWT / adaptive-DWT pilot is run with shared task episodes and written summary.

## 10. Run sequence and checkpoint schedule

Use a progressive compute budget. **Do not start a 10k-step job before profiling and tiny-overfit tests pass.**

1. Environment inspection and one π₀-FAST pretrained eval smoke episode.
2. Build codec, tokenizer, DP oracle, test suite; offline compare action reconstruction and token rate on >=1,000 disjoint action windows.
3. Train a tiny DWT model on 32–128 action windows and verify overfit + constrained decoding.
4. Benchmark 100–200 warmed-up training steps on the A100, log sec/step and VRAM, choose largest stable microbatch, then configure gradient accumulation.
5. Train each of `fast_matched`, `dwt_fixed`, `dwt_adaptive` to 2k, 5k, 10k optimizer steps with checkpoints and validation curves. If 2k shows no token-learning signal, debug before extending.
6. Evaluate each saved model on exactly matched LIBERO initial states/seeds; save videos and JSON results.
7. Summarize results, failure cases and whether adaptive token count decreased **without statistically obvious loss of task success**.
8. Only then scale to more tasks and 32-step horizons; run at least 3 random seeds for promising configurations if budget allows.

**Runtime expectations are hypotheses, not promises.** Pilot fine-tuning might be hours on a single A100, but sequence length, full vocabulary logits, image token counts, gradient checkpointing, decoder implementation and simulator throughput strongly affect elapsed time. **Always extrapolate from an actual 100–200-step profile.** For A6000 use the same gated sequence with smaller microbatch and more accumulation.

## 11. Agent deliverables

At completion, produce:

1. Runnable git repository/working directory implementing custom policy, model and codec; documented environment and pinned dependency versions.
2. `README.md` with tested install/train/eval commands for A100 and A6000, explicitly noting commands not yet tested.
3. Passing unit/integration tests with test output and actual GPU/CPU device used.
4. Example encoded action token stream, grammar trace and inverse-DWT trajectory plot.
5. `results/codec_metrics.csv`, Pareto plots, and a short `results/offline_codec_report.md`.
6. Logs/checkpoints (or paths) for tiny overfit and at least one real pilot training run, with actual elapsed time and peak VRAM.
7. `results/eval.json`, optional videos, per-task success counts and inference latency for executed rollouts.
8. `results/summary.md` with side-by-side comparison against FAST and fixed DWT, experimental caveats, and what to do next.

**Minimum successful handoff:** working codec tests + a loading/training/inference-compatible custom LeRobot policy + one small real GPU training run + one real LIBERO rollout + reproducible commands. Never fabricate missing artifacts; explicitly state blockers.

## 12. Known scientific/engineering risks (do not ignore)

- **New vocabulary / output head:** LeRobot's PI0Fast model may require explicit resizing or replacement of custom embedding/head modules. Check ties, shape consistency, checkpoint compatibility and optimizer inclusion.
- **Training objective alignment:** verify the first action token is included in teacher-forced loss. Use a tiny hand-checkable toy model/trace.
- **Quantizer generalization:** per-window normalization cannot be used unless scaling is transmitted in the token stream. Calibrate on training data only.
- **Adaptive oracle != learned selector:** an oracle that looks at future ground-truth actions can establish representation potential but does not imply the policy can make the right choices online.
- **Short-horizon costs:** two control tokens per segment, gripper tokens and variable DWT coefficients can easily cost *more* tokens than FAST+BPE, especially at H=10. Test compression, do not presume improvement.
- **Dynamic resolution is not dynamic sensing:** predicting more wavelet detail does not generate additional image observations; replanning/observation cadence is a separate control.
- **Wavelet boundary artifacts:** compare per-segment reconstruction velocity/jerk and enforce continuity where appropriate; avoid masking real sharp transitions.
- **Distribution shift:** small action reconstruction errors may cause compounding drift in long-horizon closed-loop manipulation.
- **Speed:** lower token count helps only if autoregressive inference is optimized, ideally with KV caching. Report *actual end-to-end latency*.
- **Baseline equality:** if custom policy fine-tunes new head while pretrained FAST starts with mature action embeddings, clearly disclose adaptation difficulty and run matched fresh-representation baselines where possible.
- **Gripper fidelity:** inspect state conventions; exact binary gripper event encoding is critical for manipulation.

## 13. Immediate instruction to the coding agent

**Start by** checking out LeRobot at a pinned commit, recording available GPU memory, inspecting the π₀-FAST action-token training/inference code and current `lerobot/libero` schema. Next, produce the codec with exhaustive unit tests and a FAST-vs-DWT offline comparison. Do not spend significant GPU time until the codec passes these tests. Then implement the custom plugin, prove tiny-set overfitting, profile the A100, and run a small closed-loop LIBERO evaluation. Keep the user-facing progress report factual and include exact measured commands/results.

---

### Source references (for engineering verification)

- https://github.com/huggingface/lerobot
- https://huggingface.co/docs/lerobot/pi0fast
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/processor_pi0_fast.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/configuration_pi0_fast.py
- https://huggingface.co/docs/lerobot/main/en/bring_your_own_policies
- https://huggingface.co/docs/lerobot/main/en/libero
- https://huggingface.co/datasets/lerobot/libero
- https://github.com/Physical-Intelligence/openpi

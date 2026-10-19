"""Scope 4: Launch resumable VLA fine-tuning and fit/inspect spectral tokenization statistics."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from .spectrum import create_spectrum_module, dct_ii, parse_freq_band


TRAINING_MODES = ("direct", "fast_dct", "spectral_filtered", "custom_hybrid")


def fit_spectral_dataset_stats(
    raw_dataset_dir: Path,
    schemas: Sequence[str] = ("fast_idct", "band_idct:0.35", "ifft:0.35", "idwt_haar", "custom_hybrid"),
    chunk_size: int = 10,
    quant_scale: float = 20.0,
) -> dict[str, Any]:
    """Analyze a raw Golden Dataset's 10-step action chunks across frequency schemas.

    Computes per-mode DCT energy decay, reconstruction MSE, high-frequency
    reduction, and active token compression ratio for every schema.
    """
    episode_dirs = sorted(raw_dataset_dir.glob("episode_*"))
    if not episode_dirs:
        raise ValueError(f"No episodes found in {raw_dataset_dir}")

    all_chunks = []
    robot = "so101"
    fps = 30
    for ep_dir in episode_dirs:
        meta_path = ep_dir / "metadata.json"
        frames_path = ep_dir / "frames.npz"
        if not meta_path.exists() or not frames_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("split", "train") != "train" and len(episode_dirs) > 5:
            continue
        robot = str(meta.get("robot", robot))
        fps = int(meta.get("fps", fps))
        with np.load(frames_path) as archive:
            actions = archive["actions"].astype(float)
        for start in range(0, len(actions), chunk_size):
            window = actions[start : start + chunk_size]
            if len(window) < chunk_size:
                pad = np.repeat(window[-1:, :], chunk_size - len(window), axis=0)
                window = np.vstack([window, pad])
            all_chunks.append(window)

    if not all_chunks:
        raise ValueError("No valid training action chunks found")

    stacked = np.stack(all_chunks, axis=0)  # (M, H, D)
    centered_dct = np.stack([dct_ii(c - c[0:1, :]) for c in stacked], axis=0)
    mode_energy = np.mean(centered_dct**2, axis=(0, 2))
    total_energy = float(np.sum(mode_energy)) or 1.0
    normalized_mode_energy = (mode_energy / total_energy).tolist()

    schema_stats: dict[str, Any] = {}
    for schema_spec in schemas:
        mod = create_spectrum_module(schema_spec, quant_scale=quant_scale)
        recon_errors = []
        non_zero_ratios = []
        arm_roughness_before = []
        arm_roughness_after = []
        for idx, chunk in enumerate(stacked):
            prev_tail = stacked[idx - 1][-2:] if idx > 0 else None
            encoded = mod.encode(chunk)
            reconstructed = mod.decode(encoded, prev_chunk_tail=prev_tail)
            recon_errors.append(float(np.mean((reconstructed - chunk) ** 2)))
            non_zero_ratios.append(float(encoded.get("non_zero_ratio", 1.0)))
            # Second-difference roughness on arm joints (all except gripper)
            arm_before = chunk[:, :-1] if chunk.shape[1] > 1 else chunk
            arm_after = reconstructed[:, :-1] if reconstructed.shape[1] > 1 else reconstructed
            if len(arm_before) >= 3:
                d2_before = np.diff(arm_before, n=2, axis=0)
                d2_after = np.diff(arm_after, n=2, axis=0)
                arm_roughness_before.append(float(np.mean(d2_before**2)))
                arm_roughness_after.append(float(np.mean(d2_after**2)))

        rough_before = float(np.mean(arm_roughness_before)) if arm_roughness_before else 1.0
        rough_after = float(np.mean(arm_roughness_after)) if arm_roughness_after else 1.0
        schema_stats[schema_spec] = {
            "schema_name": mod.schema_name,
            "freq_band": list(mod.freq_band),
            "fast_tokenization": mod.fast_tokenization,
            "mean_reconstruction_mse": float(np.mean(recon_errors)),
            "mean_non_zero_token_ratio": float(np.mean(non_zero_ratios)),
            "high_freq_roughness_reduction_ratio": (
                1.0 - (rough_after / rough_before) if rough_before > 1e-12 else 0.0
            ),
        }

    return {
        "robot": robot,
        "fps": fps,
        "chunk_size": chunk_size,
        "num_chunks_analyzed": len(stacked),
        "action_mean": np.mean(stacked, axis=(0, 1)).tolist(),
        "action_std": np.std(stacked, axis=(0, 1)).tolist(),
        "dct_mode_energy_distribution": normalized_mode_energy,
        "schemas": schema_stats,
    }


def training_command(
    dataset_repo: str,
    output: Path,
    steps: int,
    resume: bool,
    dataset_root: Optional[Path] = None,
    device: str = "mps",
    config_path: Optional[Path] = None,
    robot: str = "scara",
) -> list[str]:
    sibling = Path(sys.executable).with_name("lerobot-train")
    executable = str(sibling) if sibling.exists() else shutil.which("lerobot-train")
    if executable is None:
        raise RuntimeError(
            "Training requires Python 3.12 and `pip install 'lerobot[smolvla,training]'`"
        )
    command = [
        executable,
        "--policy.path=lerobot/smolvla_base",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={dataset_repo}",
        "--dataset.eval_split=0.0",
        "--batch_size=1",
        "--num_workers=0",
        "--persistent_workers=false",
        "--policy.chunk_size=10",
        "--policy.n_action_steps=10",
        f"--steps={steps}",
        f"--output_dir={output}",
        "--job_name=smooth_operator_smolvla",
        "--wandb.enable=false",
        "--eval_steps=0",
        "--save_freq=1000",
    ]
    if robot == "scara":
        command.append(
            '--rename_map={"observation.images.workspace":"observation.images.camera1"}'
        )
    if dataset_root is not None:
        command.append(f"--dataset.root={dataset_root}")
    if resume:
        command.append("--resume=true")
        if config_path is None:
            raise ValueError("--config-path is required when resuming training")
        command.append(f"--config_path={config_path}")
    return command


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "mps":
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dataset_robot(dataset_root: Optional[Path], requested: str, repo_id: str) -> str:
    if requested != "auto":
        return requested
    if dataset_root is not None:
        manifest = dataset_root.parent / "conversion.json"
        if manifest.exists():
            return str(json.loads(manifest.read_text()).get("robot", "scara"))
    return "so101" if "so101" in repo_id.lower() else "scara"


def resolve_training_spectrum_schema(
    training_mode: str,
    spectrum_schema: str,
    freq_band: str,
) -> tuple[str, tuple[float, float]]:
    band = parse_freq_band(freq_band)
    if training_mode == "direct":
        return "direct", (0.0, 1.0)
    if training_mode == "fast_dct":
        return "fast_idct", band
    if training_mode == "custom_hybrid":
        default_band = band if band != (0.0, 1.0) else (0.0, 0.35)
        return "custom_hybrid", default_band
    if training_mode == "spectral_filtered":
        schema = spectrum_schema if spectrum_schema != "direct" else "band_idct"
        default_band = band if band != (0.0, 1.0) else (0.0, 0.35)
        return schema, default_band
    raise ValueError(f"Unsupported training_mode {training_mode!r}")


def fit_spectrum_main() -> None:
    parser = argparse.ArgumentParser(description="Fit and inspect spectral tokenization stats on Golden data.")
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Path to raw Golden episodes directory")
    parser.add_argument(
        "--schemas",
        default="fast_idct,band_idct:0.35,ifft:0.35,idwt_haar,idwt_db4,custom_hybrid",
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--quant-scale", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/spectral_analysis/spectrum_stats.json"))
    args = parser.parse_args()
    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]
    stats = fit_spectral_dataset_stats(
        args.raw_dataset,
        schemas=schemas,
        chunk_size=args.chunk_size,
        quant_scale=args.quant_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--robot", choices=("auto", "scara", "so101"), default="auto")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="direct")
    parser.add_argument("--spectrum-schema", default="direct")
    parser.add_argument("--freq-band", default="0.0,1.0")
    args = parser.parse_args()
    device = resolve_device(args.device)
    robot = dataset_robot(args.dataset_root, args.robot, args.dataset_repo)
    resolved_schema, resolved_band = resolve_training_spectrum_schema(
        args.training_mode, args.spectrum_schema, args.freq_band
    )
    command = training_command(
        args.dataset_repo,
        args.output,
        args.steps,
        args.resume,
        args.dataset_root,
        device,
        args.config_path,
        robot,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "dataset_repo": args.dataset_repo,
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "steps": args.steps,
        "device": device,
        "robot": robot,
        "training_mode": args.training_mode,
        "spectrum_schema": resolved_schema,
        "freq_band": list(resolved_band),
        "resumed": args.resume,
        "started_at": time.time(),
    }
    run_path = args.output.parent / f"{args.output.name}_run.json"
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    subprocess.run(command, check=True)
    run["finished_at"] = time.time()
    run["elapsed_s"] = run["finished_at"] - run["started_at"]
    run_path.write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()

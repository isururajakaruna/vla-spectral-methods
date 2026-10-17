"""Launch resumable SmolVLA fine-tuning through LeRobot."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional


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
    args = parser.parse_args()
    device = resolve_device(args.device)
    robot = dataset_robot(args.dataset_root, args.robot, args.dataset_repo)
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

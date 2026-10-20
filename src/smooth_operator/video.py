"""MP4 encoding shared by the simulator studio and the VLA inference process."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def encode_mp4_video(frames: list[np.ndarray], out_path: Path, fps: int = 12) -> bool:
    """Encode a list of RGB24 uint8 frames into a browser-compatible H.264 MP4 video via ffmpeg."""
    if not frames:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        raw_bytes = b"".join(np.ascontiguousarray(f, dtype=np.uint8).tobytes() for f in frames)
        subprocess.run(cmd, input=raw_bytes, check=True, timeout=60.0)
        return out_path.exists()
    except Exception:
        return False

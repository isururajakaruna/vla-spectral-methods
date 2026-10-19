"""Interactive Web Control Center for Smooth Operator VLA & Spectral Experiments.

Provides a browser-based studio with:
1. Model & Control Domain Selector (explicitly separated into Time-Domain Direct
   vs. Frequency-Domain Spectral Reconstruction Models + cutoff & quantization controls).
2. Robotic Arm & Scene Arrangement Selector (SO-101 6-DoF vs. SCARA 3-DoF + seed presets)
   with live rendered PyBullet multi-camera previews, 2D workspace schematic showing
   robot base, object coordinates, obstacles, and camera frustums/placements.
3. Pre-configured Prompt Bank tailored to the selected robot arrangement + Custom Prompt input.
4. "Launch Live Simulation" button that spawns the real-time 3D PyBullet GUI window
   (and/or streams live telemetry & 5-panel spectral diagnostic charts back to the browser).
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

from .metrics import save_spectral_diagnostic_chart
from .motion import DESTINATION_XY as SCARA_DEFAULT_DEST
from .pipeline import (
    ModularExecutionPipeline,
    PrecomputedChunkPolicy,
    apply_scene_spec,
    generate_oracle_action_sequence,
    resolve_scene_from_prompt,
)
from .planning import WORKSPACE as SCARA_WORKSPACE
from .simulation import ManipulatorSimulation
from .so101 import SO101AStarController, SO101Simulation
from .spectrum import create_spectrum_module, parse_freq_band


ARRANGEMENT_CATALOG: dict[str, dict[str, Any]] = {
    "so101_seed_9": {
        "id": "so101_seed_9",
        "title": "SO-101 6-DoF Arm — Corridor Setup A (Seed 9)",
        "robot": "so101",
        "seed": 9,
        "badge": "6-DoF Official SO-ARM100 | 30 Hz",
        "description": (
            "Official 6-DoF SO-101 manipulator mounted at (-0.21m, 0.00m) with a 2-finger moving jaw gripper. "
            "Features 2 randomized wooden corridor obstacles on a 48x40cm table and 3 calibrated RGB cameras."
        ),
        "cameras": [
            {
                "id": "camera1",
                "name": "Camera 1 — Overhead Top-Down",
                "eye": [0.0, 0.0, 0.62],
                "target": [0.0, 0.0, 0.0],
                "fov_deg": 60,
                "role": "Global 2D workspace & obstacle layout",
            },
            {
                "id": "camera2",
                "name": "Camera 2 — Wrist Eye-in-Hand",
                "eye": "Dynamic (mounted on gripper_frame_link looking along +Z grasp axis)",
                "target": "Tool grasp axis (+14cm through gripper jaws)",
                "fov_deg": 60,
                "role": "Close-range jaw alignment & grasp verification",
            },
            {
                "id": "camera3",
                "name": "Camera 3 — Isometric Side Angle",
                "eye": [0.36, -0.38, 0.28],
                "target": [-0.02, 0.0, 0.05],
                "fov_deg": 60,
                "role": "3D vertical lift & obstacle clearance depth view",
            },
        ],
        "prompts": [
            {
                "label": "Target Goal: Move Apple",
                "prompt": "Move the apple to the target",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Target Goal: Carry Coke Can",
                "prompt": "Carry the Coke can to the destination",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Target Goal: Transfer Pen",
                "prompt": "Transfer the pen into the goal",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Spatial Relation: Apple Right of Coke Can",
                "prompt": "Place the apple to the right of the Coke can",
                "category": "Relational Grounding",
            },
            {
                "label": "Spatial Relation: Pen In Front of Apple",
                "prompt": "Put the pen in front of the apple",
                "category": "Relational Grounding",
            },
            {
                "label": "Spatial Relation: Apple Left of Pen",
                "prompt": "Move the apple to the left of the pen",
                "category": "Relational Grounding",
            },
        ],
    },
    "so101_seed_42": {
        "id": "so101_seed_42",
        "title": "SO-101 6-DoF Arm — Slalom Setup B (Seed 42)",
        "robot": "so101",
        "seed": 42,
        "badge": "6-DoF Official SO-ARM100 | 30 Hz",
        "description": (
            "SO-101 6-DoF setup with central obstacle placement requiring multi-waypoint A* detours "
            "around obstacles while maintaining low-height jaw clearance."
        ),
        "cameras": [
            {
                "id": "camera1",
                "name": "Camera 1 — Overhead Top-Down",
                "eye": [0.0, 0.0, 0.62],
                "target": [0.0, 0.0, 0.0],
                "fov_deg": 60,
                "role": "Global 2D workspace & obstacle layout",
            },
            {
                "id": "camera2",
                "name": "Camera 2 — Wrist Eye-in-Hand",
                "eye": "Dynamic (mounted on gripper_frame_link looking along +Z grasp axis)",
                "target": "Tool grasp axis (+14cm through gripper jaws)",
                "fov_deg": 60,
                "role": "Close-range jaw alignment & grasp verification",
            },
            {
                "id": "camera3",
                "name": "Camera 3 — Isometric Side Angle",
                "eye": [0.36, -0.38, 0.28],
                "target": [-0.02, 0.0, 0.05],
                "fov_deg": 60,
                "role": "3D vertical lift & obstacle clearance depth view",
            },
        ],
        "prompts": [
            {
                "label": "Target Goal: Move Apple",
                "prompt": "Move the apple to the target",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Target Goal: Carry Coke Can",
                "prompt": "Carry the Coke can to the destination",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Spatial Relation: Coke Can Behind Apple",
                "prompt": "Place the Coke can behind the apple",
                "category": "Relational Grounding",
            },
            {
                "label": "Spatial Relation: Apple Right of Pen",
                "prompt": "Put the apple to the right of the pen",
                "category": "Relational Grounding",
            },
        ],
    },
    "so101_canonical": {
        "id": "so101_canonical",
        "title": "SO-101 6-DoF Arm — Canonical Benchmark Grid (Static)",
        "robot": "so101",
        "seed": None,
        "badge": "6-DoF Official SO-ARM100 | 30 Hz",
        "description": (
            "Canonical unseeded SO-101 benchmark layout with 3 fixed symmetric obstacles and "
            "deterministic object starting cells."
        ),
        "cameras": [
            {
                "id": "camera1",
                "name": "Camera 1 — Overhead Top-Down",
                "eye": [0.0, 0.0, 0.62],
                "target": [0.0, 0.0, 0.0],
                "fov_deg": 60,
                "role": "Global 2D workspace & obstacle layout",
            },
            {
                "id": "camera2",
                "name": "Camera 2 — Wrist Eye-in-Hand",
                "eye": "Dynamic (mounted on gripper_frame_link looking along +Z grasp axis)",
                "target": "Tool grasp axis (+14cm through gripper jaws)",
                "fov_deg": 60,
                "role": "Close-range jaw alignment & grasp verification",
            },
            {
                "id": "camera3",
                "name": "Camera 3 — Isometric Side Angle",
                "eye": [0.36, -0.38, 0.28],
                "target": [-0.02, 0.0, 0.05],
                "fov_deg": 60,
                "role": "3D vertical lift & obstacle clearance depth view",
            },
        ],
        "prompts": [
            {
                "label": "Target Goal: Move Apple",
                "prompt": "Move the apple to the target",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Target Goal: Carry Coke Can",
                "prompt": "Carry the Coke can to the destination",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Target Goal: Transfer Pen",
                "prompt": "Transfer the pen into the goal",
                "category": "Goal Marker Transport",
            },
        ],
    },
    "scara_seed_9": {
        "id": "scara_seed_9",
        "title": "SCARA 3-DoF Planar Arm — Wide Table Setup (Seed 9)",
        "robot": "scara",
        "seed": 9,
        "badge": "3-DoF Planar SCARA | 10 Hz",
        "description": (
            "Three-joint SCARA manipulator (shoulder, elbow, vertical prismatic lift) mounted at "
            "(-0.68m, 0.00m) with a 1.45m reach across a 1.2x1.0m grid table and 1 global workspace camera."
        ),
        "cameras": [
            {
                "id": "workspace",
                "name": "Workspace Camera — Elevated Isometric",
                "eye": [0.90, -1.00, 1.35],
                "target": [0.00, 0.00, 0.15],
                "fov_deg": 55,
                "role": "Full-table RGB observation for SCARA VLA policy",
            },
        ],
        "prompts": [
            {
                "label": "Target Goal: Move Apple",
                "prompt": "Move the apple to the target",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Target Goal: Carry Coke Can",
                "prompt": "Carry the Coke can to the destination",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Target Goal: Transfer Pen",
                "prompt": "Transfer the pen into the goal",
                "category": "Goal Marker Transport",
            },
            {
                "label": "Spatial Relation: Apple Right of Coke Can",
                "prompt": "Place the apple to the right of the Coke can",
                "category": "Relational Grounding",
            },
            {
                "label": "Spatial Relation: Pen In Front of Apple",
                "prompt": "Put the pen in front of the apple",
                "category": "Relational Grounding",
            },
        ],
    },
}


MODEL_CATALOG: dict[str, list[dict[str, Any]]] = {
    "time_domain": [
        {
            "id": "td_astar_direct",
            "name": "A* Oracle Policy (Direct Time-Domain)",
            "model": "astar_oracle",
            "version": "SO101-AStar v3.0 (Exact IK + Collision Corridor)",
            "checkpoint": "built-in/astar_oracle_v3.0",
            "fast_tokenization": False,
            "spectrum_schema": "direct",
            "freq_band": "0.0,1.0",
            "badge": "A* Oracle v3.0 | Direct TD",
            "description": (
                "Executes raw 10-step joint position action chunks directly in the time domain "
                "without spectral decomposition or inter-chunk smoothing (fast_tokenization=False)."
            ),
        },
        {
            "id": "td_smolvla_direct",
            "name": "SmolVLA Pretrained / Fine-Tuned (Direct Time-Domain)",
            "model": "smolvla",
            "version": "SmolVLA-Base v0.4.3 (lerobot/smolvla_base | 450M)",
            "checkpoint": "lerobot/smolvla_base",
            "fast_tokenization": False,
            "spectrum_schema": "direct",
            "freq_band": "0.0,1.0",
            "badge": "SmolVLA v0.4.3 | Direct TD",
            "description": (
                "Queries the SmolVLA Vision-Language-Action model over TCP socket and applies "
                "predicted continuous time-domain action chunks directly to the robot."
            ),
        },
    ],
    "frequency_domain": [
        {
            "id": "fd_custom_hybrid",
            "name": "Proposed Custom Hybrid Spectrum (Channel-Decoupled Butterworth DCT + C1 Blend)",
            "model": "astar_oracle",
            "version": "CustomHybrid-DCT v2.1 (Butterworth + C1 Blend)",
            "checkpoint": "smooth-operator/custom_hybrid_v2.1",
            "fast_tokenization": True,
            "spectrum_schema": "custom_hybrid",
            "freq_band": "0.0,0.35",
            "badge": "CustomHybrid v2.1 (Ours)",
            "description": (
                "Decouples arm transport joints (3rd-order Butterworth DCT low-pass filtering) from "
                "the discrete gripper jaw channel (step-preserving trace) and applies C1 cosine-squared "
                "velocity blending across 10-step chunk boundaries."
            ),
        },
        {
            "id": "fd_fast_idct",
            "name": "Standard FAST Tokenization (Full-Band Orthonormal DCT-II / IDCT-III)",
            "model": "astar_oracle",
            "version": "FAST-DCT v1.0 (Physical Intelligence Spec)",
            "checkpoint": "fast-tokenizer/dct_ii_v1.0",
            "fast_tokenization": True,
            "spectrum_schema": "fast_idct",
            "freq_band": "0.0,1.0",
            "badge": "FAST-DCT v1.0 | Full-Band",
            "description": (
                "Encodes 10-step action chunks into quantized orthonormal DCT-II tokens and reconstructs "
                "via full-band Type-III Inverse DCT."
            ),
        },
        {
            "id": "fd_band_idct",
            "name": "Band-Limited IDCT (Low-Pass Frequency Truncated DCT)",
            "model": "astar_oracle",
            "version": "BandIDCT v1.2 (Butterworth Spectral Window)",
            "checkpoint": "smooth-operator/band_idct_v1.2",
            "fast_tokenization": True,
            "spectrum_schema": "band_idct",
            "freq_band": "0.0,0.35",
            "badge": "Band-IDCT v1.2 | Low-Pass",
            "description": (
                "Attenuates high-frequency DCT modes above the normalized cutoff frequency using a "
                "Butterworth spectral window before IDCT reconstruction."
            ),
        },
        {
            "id": "fd_ifft",
            "name": "Inverse Fast Fourier Transform (Trend-Removed RFFT / IRFFT)",
            "model": "astar_oracle",
            "version": "IRFFT-Detrend v1.1 (Harmonic Fourier)",
            "checkpoint": "smooth-operator/ifft_v1.1",
            "fast_tokenization": True,
            "spectrum_schema": "ifft",
            "freq_band": "0.0,0.40",
            "badge": "IRFFT v1.1 | Fourier Basis",
            "description": (
                "Removes linear chunk endpoint trend, filters complex RFFT harmonics in the frequency "
                "domain, and reconstructs via Inverse Real FFT."
            ),
        },
        {
            "id": "fd_idwt_haar",
            "name": "Haar Discrete Wavelet Reconstruction (2-Level Periodic IDWT)",
            "model": "astar_oracle",
            "version": "Haar-DWT v1.0 (2-Level Multi-Resolution)",
            "checkpoint": "smooth-operator/idwt_haar_v1.0",
            "fast_tokenization": True,
            "spectrum_schema": "idwt_haar",
            "freq_band": "0.0,0.50",
            "badge": "Haar-IDWT v1.0 | Wavelet",
            "description": (
                "Multi-resolution 2-level Haar wavelet decomposition with soft-thresholding on "
                "high-frequency detail coefficients."
            ),
        },
        {
            "id": "fd_idwt_db4",
            "name": "Daubechies-4 Wavelet Reconstruction (2-Level Periodic DB4 IDWT)",
            "model": "astar_oracle",
            "version": "Daubechies-DB4 v1.0 (4-Tap Orthogonal Wavelet)",
            "checkpoint": "smooth-operator/idwt_db4_v1.0",
            "fast_tokenization": True,
            "spectrum_schema": "idwt_db4",
            "freq_band": "0.0,0.50",
            "badge": "DB4-IDWT v1.0 | Wavelet",
            "description": (
                "Smooth 4-tap Daubechies orthogonal wavelet filter bank with multi-scale detail shrinkage."
            ),
        },
        {
            "id": "fd_smolvla_hybrid",
            "name": "SmolVLA + Proposed Custom Hybrid Spectrum Decoder",
            "model": "smolvla",
            "version": "SmolVLA v0.4.3 + CustomHybrid v2.1",
            "checkpoint": "lerobot/smolvla_base + custom_hybrid_v2.1",
            "fast_tokenization": True,
            "spectrum_schema": "custom_hybrid",
            "freq_band": "0.0,0.35",
            "badge": "SmolVLA v0.4.3 + Hybrid v2.1",
            "description": (
                "Combines neural SmolVLA action chunk generation with our Channel-Decoupled + C1 "
                "Boundary-Blended Inverse Spectrum Module."
            ),
        },
    ],
}


def get_gpu_hardware_status() -> dict[str, Any]:
    """Detect NVIDIA GPU hardware, PyBullet EGL hardware OpenGL renderer, and PyTorch CUDA status."""
    import pkgutil
    import shutil

    gpu_name = "CPU Only"
    driver_version = "N/A"
    vram_mb = "N/A"
    has_nvidia = False
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3.0,
            ).strip()
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                if len(parts) >= 3:
                    gpu_name, driver_version, vram_mb = parts[0], parts[1], f"{parts[2]} MiB"
                    has_nvidia = True
        except Exception:
            pass

    egl_available = pkgutil.get_loader("eglRenderer") is not None
    torch_cuda = False
    torch_version = "Not installed (Python 3.10 env)"
    try:
        import torch  # type: ignore

        torch_version = str(torch.__version__)
        torch_cuda = bool(torch.cuda.is_available())
    except Exception:
        pass

    return {
        "has_nvidia_gpu": has_nvidia,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "vram_total": vram_mb,
        "pybullet_egl_gpu_rendering": egl_available and has_nvidia,
        "torch_version": torch_version,
        "torch_cuda_available": torch_cuda,
    }


def _np_to_base64_png(rgb_array: np.ndarray) -> str:
    img = Image.fromarray(np.asarray(rgb_array, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_schematic_diagram(
    robot: str,
    workspace: tuple[float, float, float, float],
    base_xy: tuple[float, float],
    objects: dict[str, list[float]],
    obstacles: list[list[float]],
    goal_xy: tuple[float, float],
) -> str:
    """Draw a clean Google-style light 2D top-down engineering schematic showing objects, obstacles, arm base, and cameras."""
    w, h = 520, 420
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_l, margin_r, margin_t, margin_b = 52, 480, 42, 370
    draw.rectangle((margin_l, margin_t, margin_r, margin_b), fill=(248, 249, 250), outline=(218, 220, 224), width=2)
    draw.text((margin_l, 14), f"2D Workspace & Camera Placement Schematic ({robot.upper()})", fill=(32, 33, 36))

    xmin, ymin, xmax, ymax = workspace

    def proj(pt: tuple[float, float] | list[float]) -> tuple[float, float]:
        px = margin_l + (float(pt[0]) - xmin) / max(xmax - xmin, 1e-6) * (margin_r - margin_l)
        py = margin_b - (float(pt[1]) - ymin) / max(ymax - ymin, 1e-6) * (margin_b - margin_t)
        return (px, py)

    # Grid lines
    for gx in np.linspace(xmin, xmax, 7):
        p0, p1 = proj((gx, ymin)), proj((gx, ymax))
        draw.line([p0, p1], fill=(232, 234, 237), width=1)
        draw.text((p0[0] - 14, margin_b + 6), f"{gx:+.2f}m", fill=(95, 99, 104))
    for gy in np.linspace(ymin, ymax, 6):
        p0, p1 = proj((xmin, gy)), proj((xmax, gy))
        draw.line([p0, p1], fill=(232, 234, 237), width=1)
        draw.text((8, p0[1] - 6), f"{gy:+.2f}m", fill=(95, 99, 104))

    # Obstacles
    for idx, obs in enumerate(obstacles):
        tl = proj((obs[0], obs[3]))
        br = proj((obs[2], obs[1]))
        draw.rectangle((*tl, *br), fill=(254, 239, 195), outline=(249, 171, 0), width=2)
        cx, cy = (tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2
        draw.text((cx - 16, cy - 6), f"Obs {idx+1}", fill=(176, 96, 0))

    # Goal marker
    gx, gy = proj(goal_xy)
    draw.rectangle((gx - 11, gy - 11, gx + 11, gy + 11), fill=(230, 244, 234), outline=(30, 142, 62), width=2)
    draw.text((gx + 14, gy - 6), f"Goal ({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})", fill=(19, 115, 51))

    # Robot base
    bx, by = proj(base_xy)
    bx_clamped = min(max(bx, margin_l + 12), margin_r - 12)
    by_clamped = min(max(by, margin_t + 12), margin_b - 12)
    draw.ellipse((bx_clamped - 11, by_clamped - 11, bx_clamped + 11, by_clamped + 11), fill=(26, 115, 232), outline=(210, 227, 252), width=2)
    draw.text((bx_clamped + 14, by_clamped - 8), f"Arm Base ({base_xy[0]:+.2f}, {base_xy[1]:+.2f})", fill=(23, 78, 166))

    # Objects
    obj_colors = {
        "apple": ((217, 48, 37), "Apple"),
        "coke_can": ((227, 116, 0), "Coke Can"),
        "pen": ((24, 128, 56), "Pen"),
    }
    for name, xyz in objects.items():
        ox, oy = proj((xyz[0], xyz[1]))
        color, label = obj_colors.get(name, ((95, 99, 104), name))
        draw.ellipse((ox - 9, oy - 9, ox + 9, oy + 9), fill=color, outline=(255, 255, 255), width=2)
        draw.text((ox + 12, oy - 7), f"{label} ({xyz[0]:+.2f}, {xyz[1]:+.2f})", fill=color)

    # Camera icons
    if robot == "so101":
        c1x, c1y = proj((0.0, 0.0))
        draw.polygon([(c1x, c1y - 8), (c1x - 7, c1y + 6), (c1x + 7, c1y + 6)], outline=(147, 52, 230), fill=(243, 232, 253))
        draw.text((c1x - 28, c1y + 9), "Cam1 (Top 0.72m)", fill=(118, 39, 187))
        draw.text((margin_r - 160, margin_b - 36), "Cam2: Wrist Eye-in-Hand", fill=(60, 64, 67))
        draw.text((margin_r - 160, margin_b - 20), "Cam3: Side Iso (0.48,-0.48,0.34)", fill=(60, 64, 67))
    else:
        draw.text((margin_r - 185, margin_b - 22), "Cam1: Iso (0.90, -1.00, 1.35m)", fill=(60, 64, 67))

    return _pil_to_base64_png(img)


_PREVIEW_CACHE: dict[str, dict[str, Any]] = {}
_SIM_LOCK = threading.Lock()
_ACTIVE_GUI_PROCESS: Optional[subprocess.Popen] = None


def _np_to_base64_jpeg(rgb_array: np.ndarray, quality: int = 86) -> str:
    img = Image.fromarray(np.asarray(rgb_array, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_so101_wrist_camera(sim: SO101Simulation, width: int = 640, height: int = 480) -> np.ndarray:
    state = p.getLinkState(
        sim.robot_id,
        sim.tool_link_index,
        computeForwardKinematics=True,
        physicsClientId=sim.client_id,
    )
    origin = np.asarray(state[4], dtype=float)
    rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    grasp_axis = rotation[:, 2]
    wrist_up = -rotation[:, 0]
    eye = origin - 0.04 * grasp_axis - 0.04 * rotation[:, 0]
    target = origin + 0.14 * grasp_axis
    return sim._render_camera(eye.tolist(), target.tolist(), wrist_up.tolist(), width, height)


def build_arrangement_preview(arrangement_id: str) -> dict[str, Any]:
    """Render live PyBullet camera views and scene metadata for the chosen robotic arrangement."""
    if arrangement_id in _PREVIEW_CACHE:
        return _PREVIEW_CACHE[arrangement_id]
    if arrangement_id not in ARRANGEMENT_CATALOG:
        raise ValueError(f"Unknown arrangement_id {arrangement_id!r}")

    cfg = ARRANGEMENT_CATALOG[arrangement_id]
    robot = cfg["robot"]
    seed = cfg["seed"]
    sim_cls = SO101Simulation if robot == "so101" else ManipulatorSimulation

    with _SIM_LOCK:
        with sim_cls(gui=False, training_seed=seed) as sim:
            default_dest = (
                sim.DEFAULT_DESTINATION if isinstance(sim, SO101Simulation) else SCARA_DEFAULT_DEST
            )
            sim.set_goal_marker(default_dest)
            if isinstance(sim, SO101Simulation):
                # Position the SO-101 arm in a natural ready pose above the workspace center
                try:
                    ctrl = SO101AStarController(sim)
                    q_ready = ctrl.inverse_kinematics((-0.02, 0.0, 0.16))
                    q_ready[5] = ctrl.GRIPPER_OPEN
                    sim.command_joints(q_ready)
                    sim.step(60)
                except Exception:
                    pass
            objects_xyz = {
                name: [round(float(v), 4) for v in sim.object_position(body)]
                for name, body in sim.object_ids.items()
            }
            obstacles = [[round(float(v), 4) for v in b] for b in sim.obstacle_bounds]
            joint_names = [
                p.getJointInfo(sim.robot_id, idx, physicsClientId=sim.client_id)[1].decode("utf-8")
                for idx in sim.joint_indices
            ]

            camera_renders: dict[str, str] = {}
            if isinstance(sim, SO101Simulation):
                # Render all 3 calibrated cameras in crisp 640x480 with the robot arm visible
                views = {
                    "camera1": ((0.0, 0.0, 0.62), (0.0, 0.0, 0.0), (0, 1, 0)),
                    "camera3": ((0.36, -0.38, 0.28), (-0.02, 0.0, 0.05), (0, 0, 1)),
                }
                for key, (eye, target, up) in views.items():
                    rgb = sim._render_camera(eye, target, up, 640, 480)
                    camera_renders[key] = _np_to_base64_png(rgb)
                # Wrist eye-in-hand camera looking along the gripper jaws toward the workspace
                rgb_cam2 = _render_so101_wrist_camera(sim, width=640, height=480)
                camera_renders["camera2"] = _np_to_base64_png(rgb_cam2)
                workspace = SO101Simulation.WORKSPACE
                base_xy = SO101Simulation.ROBOT_BASE[:2]
            else:
                rgb = sim.camera_observation(width=640, height=480)
                camera_renders["workspace"] = _np_to_base64_png(rgb)
                workspace = SCARA_WORKSPACE
                base_xy = (-0.68, 0.0)

    schematic_b64 = _render_schematic_diagram(
        robot=robot,
        workspace=workspace,
        base_xy=base_xy,
        objects=objects_xyz,
        obstacles=obstacles,
        goal_xy=default_dest,
    )

    payload = {
        **cfg,
        "joint_names": joint_names,
        "objects": objects_xyz,
        "obstacles": obstacles,
        "default_destination": list(default_dest),
        "camera_images": camera_renders,
        "schematic_image": schematic_b64,
    }
    _PREVIEW_CACHE[arrangement_id] = payload
    return payload


RUNS_DIR = Path("data/runs")
VIDEO_FPS = 12
SCENE_VIEW_SIZE = (800, 450)
SCENE_VIEW_PRESETS: dict[str, dict[str, Any]] = {
    "so101": {"target": (-0.06, 0.0, 0.06), "distance": 0.78, "pitch": -28.0, "yaw": 35.0},
    "scara": {"target": (-0.15, 0.0, 0.12), "distance": 2.1, "pitch": -32.0, "yaw": 35.0},
}


class LiveFrameBroker:
    """Holds the latest JPEG of the running simulation for MJPEG streaming."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[bytes] = None
        self._seq = 0
        self._active = False
        self.run_id: Optional[str] = None

    def start(self, run_id: str) -> None:
        with self._cond:
            self._frame = None
            self._seq += 1
            self._active = True
            self.run_id = run_id
            self._cond.notify_all()

    def publish(self, rgb: np.ndarray) -> None:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="JPEG", quality=82)
        with self._cond:
            self._frame = buf.getvalue()
            self._seq += 1
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self._active = False
            self._seq += 1
            self._cond.notify_all()

    def wait_next(self, last_seq: int, timeout: float) -> tuple[int, Optional[bytes], bool]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._seq, self._frame, self._active


_LIVE_BROKER = LiveFrameBroker()


def _render_scene_view(sim: Any, robot: str, frame_idx: int) -> np.ndarray:
    """Render an external 3D view of the whole scene with a slowly orbiting camera."""
    preset = SCENE_VIEW_PRESETS.get(robot, SCENE_VIEW_PRESETS["so101"])
    width, height = SCENE_VIEW_SIZE
    yaw = preset["yaw"] + min(frame_idx * 0.6, 50.0)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=preset["target"],
        distance=preset["distance"],
        yaw=yaw,
        pitch=preset["pitch"],
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(50.0, width / height, 0.02, 6.0)
    rgba = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        shadow=1,
        lightDirection=(0.6, -0.4, 1.0),
        renderer=sim.camera_renderer,
        physicsClientId=sim.client_id,
    )[2]
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def _encode_mp4_video(frames: list[np.ndarray], out_path: Path, fps: int = 12) -> bool:
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
        subprocess.run(cmd, input=raw_bytes, check=True, timeout=20.0)
        return out_path.exists()
    except Exception:
        return False


def _find_model_entry(
    model_id: Optional[str],
    model_name: str,
    spectrum_schema: str,
    fast_tokenization: bool,
) -> dict[str, Any]:
    """Resolve full model metadata from MODEL_CATALOG."""
    for domain_key, items in MODEL_CATALOG.items():
        for entry in items:
            if model_id and entry["id"] == model_id:
                return {**entry, "domain": domain_key}
    for domain_key, items in MODEL_CATALOG.items():
        for entry in items:
            if (
                entry["model"] == model_name
                and entry["spectrum_schema"] == spectrum_schema
                and bool(entry["fast_tokenization"]) == bool(fast_tokenization)
            ):
                return {**entry, "domain": domain_key}
    return {
        "id": model_id or f"{model_name}_{spectrum_schema}",
        "name": f"{model_name} ({spectrum_schema})",
        "model": model_name,
        "version": f"{model_name} ({spectrum_schema})",
        "checkpoint": model_name,
        "badge": spectrum_schema.upper(),
        "domain": "frequency_domain" if fast_tokenization else "time_domain",
    }


def list_saved_runs() -> list[dict[str, Any]]:
    """Return all saved runs from data/runs sorted by timestamp folder name (newest first)."""
    if not RUNS_DIR.exists():
        return []
    runs: list[dict[str, Any]] = []
    # Sort folder names descending so newest timestamp folders appear first
    for folder in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        if not folder.is_dir():
            continue
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            runs.append(meta)
        except Exception:
            continue
    return runs


def execute_web_launch(params: dict[str, Any]) -> dict[str, Any]:
    """Run the selected model, spectrum schema, arrangement, and prompt; save multi-camera MP4 videos & metadata to data/runs."""
    global _ACTIVE_GUI_PROCESS
    arrangement_id = str(params.get("arrangement_id", "so101_seed_9"))
    cfg = ARRANGEMENT_CATALOG.get(arrangement_id, ARRANGEMENT_CATALOG["so101_seed_9"])
    robot = str(cfg["robot"])
    seed = cfg["seed"]

    model_id = str(params.get("model_id", "")) or None
    model_name = str(params.get("model", "astar_oracle"))
    fast_tokenization = bool(params.get("fast_tokenization", True))
    spectrum_schema = str(params.get("spectrum_schema", "custom_hybrid"))
    freq_band_str = str(params.get("freq_band", "0.0,0.35"))
    quant_scale = float(params.get("quant_scale", 40.0))
    prompt = str(params.get("prompt", "Move the apple to the target")).strip()
    open_gui = bool(params.get("open_gui", True))
    hold_seconds = float(params.get("hold_seconds", 6.0))

    freq_band = parse_freq_band(freq_band_str)
    active_schema = spectrum_schema if fast_tokenization else "direct"
    spectrum_mod = create_spectrum_module(
        schema=active_schema,
        freq_band=freq_band,
        quant_scale=quant_scale,
    )
    model_meta = _find_model_entry(model_id, model_name, active_schema, fast_tokenization)

    # Create timestamp-based folder in data/runs/ so runs sort chronologically
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. If open_gui is requested, spawn the non-blocking 3D PyBullet GUI window for live viewing
    gui_pid: Optional[int] = None
    cli_cmd = [
        sys.executable,
        "-m",
        "smooth_operator.pipeline",
        "--robot",
        robot,
        "--model",
        model_name,
        "--spectrum-schema",
        active_schema,
        "--freq-band",
        f"{freq_band[0]},{freq_band[1]}",
        "--quant-scale",
        str(quant_scale),
        "--prompt",
        prompt,
        "--gui",
        "--hold",
        str(hold_seconds),
    ]
    if seed is not None:
        cli_cmd.extend(["--seed", str(seed)])
    if fast_tokenization:
        cli_cmd.append("--fast-tokenization")

    if open_gui:
        if _ACTIVE_GUI_PROCESS is not None and _ACTIVE_GUI_PROCESS.poll() is None:
            try:
                _ACTIVE_GUI_PROCESS.terminate()
                _ACTIVE_GUI_PROCESS.wait(timeout=2.0)
            except Exception:
                pass
        _ACTIVE_GUI_PROCESS = subprocess.Popen(cli_cmd)
        gui_pid = _ACTIVE_GUI_PROCESS.pid

    # 2. Run a fast headless rollout in-process to capture multi-camera videos, telemetry, and diagnostic chart
    sim_cls = SO101Simulation if robot == "so101" else ManipulatorSimulation
    camera_frame_buffers: dict[str, list[np.ndarray]] = {
        cam["id"]: [] for cam in cfg["cameras"]
    }
    scene_frames: list[np.ndarray] = []

    with _SIM_LOCK:
        _LIVE_BROKER.start(run_id)
        with sim_cls(gui=False, training_seed=seed) as temp_sim:
            spec = resolve_scene_from_prompt(temp_sim, prompt, seed)

        oracle_actions, oracle_samples, oracle_len = generate_oracle_action_sequence(spec)
        spec.oracle_path_length_m = oracle_len
        oracle_xy = [row[1:3] for row in oracle_samples]

        with sim_cls(gui=False, training_seed=seed) as sim:
            apply_scene_spec(sim, spec)
            pipeline = ModularExecutionPipeline(
                simulation=sim,
                policy=PrecomputedChunkPolicy(oracle_actions, chunk_size=10),
                spectrum_module=spectrum_mod,
                model_name=model_name,
                realtime_gui=False,
            )
            orig_step_so101 = pipeline._step_so101
            orig_step_scara = pipeline._step_scara
            step_counter = [0]
            last_capture_time = [0.0]

            def _capture_all_cameras() -> None:
                if step_counter[0] % 4 == 0:
                    scene_rgb = _render_scene_view(sim, robot, len(scene_frames))
                    scene_frames.append(scene_rgb)
                    # Pace capture to the saved video frame rate so the live stream plays in real time
                    wait = (1.0 / VIDEO_FPS) - (time.monotonic() - last_capture_time[0])
                    if wait > 0:
                        time.sleep(wait)
                    last_capture_time[0] = time.monotonic()
                    _LIVE_BROKER.publish(scene_rgb)
                    if isinstance(sim, SO101Simulation):
                        cam1_rgb = sim._render_camera((0.0, 0.0, 0.62), (0.0, 0.0, 0.0), (0, 1, 0), 480, 360)
                        cam2_rgb = _render_so101_wrist_camera(sim, width=480, height=360)
                        cam3_rgb = sim._render_camera((0.36, -0.38, 0.28), (-0.02, 0.0, 0.05), (0, 0, 1), 480, 360)
                        if "camera1" in camera_frame_buffers:
                            camera_frame_buffers["camera1"].append(cam1_rgb)
                        if "camera2" in camera_frame_buffers:
                            camera_frame_buffers["camera2"].append(cam2_rgb)
                        if "camera3" in camera_frame_buffers:
                            camera_frame_buffers["camera3"].append(cam3_rgb)
                    else:
                        ws_rgb = sim.camera_observation(width=480, height=360)
                        if "workspace" in camera_frame_buffers:
                            camera_frame_buffers["workspace"].append(ws_rgb)
                step_counter[0] += 1

            def _wrapped_so101(policy_action: np.ndarray, object_name: str, steps: int) -> None:
                orig_step_so101(policy_action, object_name, steps)
                _capture_all_cameras()

            def _wrapped_scara(policy_action: np.ndarray, object_name: str, steps: int) -> None:
                orig_step_scara(policy_action, object_name, steps)
                _capture_all_cameras()

            pipeline._step_so101 = _wrapped_so101  # type: ignore[method-assign]
            pipeline._step_scara = _wrapped_scara  # type: ignore[method-assign]

            try:
                result = pipeline.execute_task(spec, max_chunks=25)
                _capture_all_cameras()
            finally:
                _LIVE_BROKER.finish()
            workspace = SO101Simulation.WORKSPACE if robot == "so101" else SCARA_WORKSPACE
            run_chart_path = run_dir / "diagnostic_chart.png"
            save_spectral_diagnostic_chart(
                path=run_chart_path,
                object_samples=pipeline.object_samples,
                joint_samples=pipeline.joint_samples,
                obstacles=sim.obstacle_bounds,
                metrics=result,
                title=f"{robot.upper()} | {model_meta['name']} | {spectrum_mod.schema_name} {spectrum_mod.freq_band} | {prompt}",
                workspace=workspace,
                oracle_xy=oracle_xy,
            )
            latest_chart_path = Path("artifacts/web_latest_run.png")
            latest_chart_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_chart_path, latest_chart_path)
            chart_b64 = (
                "data:image/png;base64,"
                + base64.b64encode(run_chart_path.read_bytes()).decode("ascii")
            )

    # Encode MP4 video files for every camera of this robot arrangement
    saved_cameras: list[dict[str, Any]] = []
    for cam in cfg["cameras"]:
        cam_id = cam["id"]
        frames = camera_frame_buffers.get(cam_id, [])
        video_filename = f"{cam_id}.mp4"
        video_path = run_dir / video_filename
        _encode_mp4_video(frames, video_path, fps=VIDEO_FPS)
        saved_cameras.append(
            {
                "id": cam_id,
                "name": cam["name"],
                "role": cam["role"],
                "fov_deg": cam["fov_deg"],
                "eye": cam["eye"],
                "video_file": video_filename,
                "video_url": f"/api/runs/{run_id}/{video_filename}",
                "frame_count": len(frames),
            }
        )

    scene_video: Optional[dict[str, Any]] = None
    if _encode_mp4_video(scene_frames, run_dir / "scene3d.mp4", fps=VIDEO_FPS):
        scene_video = {
            "id": "scene3d",
            "name": "3D Scene View",
            "role": "External orbiting view of the full robot and workspace",
            "video_file": "scene3d.mp4",
            "video_url": f"/api/runs/{run_id}/scene3d.mp4",
            "frame_count": len(scene_frames),
        }

    obj_samples = np.asarray(pipeline.object_samples, dtype=float)
    max_lift_cm = (
        float((obj_samples[:, 3].max() - obj_samples[0, 3]) * 100.0)
        if len(obj_samples) > 1
        else 0.0
    )

    compact_metrics = {
        k: v
        for k, v in result.items()
        if k not in ("trajectory", "joint_trajectory", "psd_freqs_hz", "psd_power")
    }
    compact_metrics["max_vertical_lift_cm"] = round(max_lift_cm, 2)

    cli_command_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cli_cmd)

    run_metadata: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": f"data/runs/{run_id}",
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_iso": now.isoformat(),
        "model_id": model_meta["id"],
        "model": model_name,
        "model_display_name": model_meta["name"],
        "model_version": model_meta["version"],
        "model_badge": model_meta["badge"],
        "checkpoint": model_meta["checkpoint"],
        "domain": model_meta["domain"],
        "fast_tokenization": fast_tokenization,
        "spectrum_schema": active_schema,
        "freq_band": list(freq_band),
        "quant_scale": quant_scale,
        "arrangement_id": arrangement_id,
        "arrangement_title": cfg["title"],
        "robot": robot,
        "robot_badge": cfg["badge"],
        "seed": seed,
        "prompt": prompt,
        "cli_command": cli_command_str,
        "gui_launched": open_gui,
        "gui_pid": gui_pid,
        "camera_count": len(saved_cameras),
        "cameras": saved_cameras,
        "scene_video": scene_video,
        "chart_url": f"/api/runs/{run_id}/diagnostic_chart.png",
        "metrics": compact_metrics,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    return {
        "status": "ok",
        "run": run_metadata,
        "gui_launched": open_gui,
        "gui_pid": gui_pid,
        "cli_command": cli_command_str,
        "metrics": compact_metrics,
        "diagnostic_chart": chart_b64,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VLA & Spectral Control Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto+Mono:wght@400;500&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f8f9fa;
      --panel: #ffffff;
      --panel-alt: #f8f9fa;
      --border: #dadce0;
      --border-subtle: #e8eaed;
      --text: #202124;
      --muted: #5f6368;
      --accent: #1a73e8;
      --accent-hover: #1557b0;
      --accent-soft: #e8f0fe;
      --accent-2: #9334e6;
      --accent-2-soft: #f3e8fd;
      --success: #1e8e3e;
      --success-soft: #e6f4ea;
      --warn: #f9ab00;
      --warn-soft: #fef7e0;
      --danger: #d93025;
      --danger-soft: #fce8e6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Google Sans', 'Roboto', -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }
    header {
      height: 56px;
      padding: 0 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 28px;
      height: 28px;
      border-radius: 7px;
      background: #ffffff;
      border: 1px solid var(--border);
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      padding: 5px;
      gap: 3px;
    }
    .icon-btn {
      position: relative;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      border: none;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: background 0.14s ease, color 0.14s ease;
    }
    .icon-btn:hover {
      background: #f1f3f4;
      color: var(--text);
    }
    .icon-btn svg {
      width: 20px;
      height: 20px;
    }
    .icon-btn .count-dot {
      position: absolute;
      top: 3px;
      right: 2px;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      border-radius: 999px;
      background: var(--accent);
      color: #ffffff;
      font-size: 10px;
      font-weight: 600;
      line-height: 16px;
      text-align: center;
    }
    .header-status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border-subtle);
      background: var(--panel-alt);
      white-space: nowrap;
    }
    .header-status .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--success);
    }
    .header-status .dot.busy {
      background: var(--warn);
    }
    .header-status .dot.cpu {
      background: var(--muted);
    }
    .brand-dot {
      border-radius: 50%;
    }
    .brand-dot.b { background: #4285F4; }
    .brand-dot.r { background: #EA4335; }
    .brand-dot.y { background: #FBBC05; }
    .brand-dot.g { background: #34A853; }
    .brand h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 500;
      color: var(--text);
      letter-spacing: -0.01em;
    }
    .status-pill {
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 500;
      background: var(--success-soft);
      color: #137333;
      border: 1px solid #ceead6;
    }
    .container {
      max-width: 1580px;
      margin: 0 auto;
      padding: 24px 28px 48px;
      display: grid;
      grid-template-columns: 600px 1fr;
      gap: 26px;
    }
    @media (max-width: 1180px) {
      .container {
        grid-template-columns: 1fr;
      }
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 20px;
      box-shadow: 0 1px 2px 0 rgba(60, 64, 67, 0.1), 0 1px 3px 1px rgba(60, 64, 67, 0.06);
    }
    .step-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }
    .step-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 15.5px;
      font-weight: 500;
      color: var(--text);
    }
    .step-num {
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: var(--accent);
      color: #ffffff;
      font-size: 13px;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .domain-tabs {
      display: inline-flex;
      gap: 2px;
      background: #f1f3f4;
      padding: 3px;
      border-radius: 999px;
      margin-bottom: 12px;
    }
    .domain-tab {
      padding: 5px 14px;
      border-radius: 999px;
      border: none;
      background: transparent;
      color: var(--muted);
      font-weight: 500;
      font-size: 12.5px;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.16s ease;
    }
    .domain-tab:hover {
      color: var(--text);
    }
    .domain-tab.active {
      background: #ffffff;
      color: var(--accent);
      box-shadow: 0 1px 2px rgba(60, 64, 67, 0.2);
    }
    .model-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 300px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .model-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 8px 9px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #ffffff;
      cursor: pointer;
      transition: all 0.14s ease;
    }
    .model-item:hover {
      border-color: #8ab4f8;
      background: #fafbfe;
    }
    .model-item.selected {
      border-color: var(--accent);
      background: #f8fbff;
      box-shadow: inset 0 0 0 1px var(--accent);
    }
    .model-radio {
      flex: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid var(--border);
      transition: all 0.14s ease;
    }
    .model-item.selected .model-radio {
      border-color: var(--accent);
      box-shadow: inset 0 0 0 3px #ffffff;
      background: var(--accent);
    }
    .model-item-body {
      flex: 1;
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .model-name {
      font-size: 13px;
      font-weight: 500;
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .model-info-btn {
      flex: none;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border: none;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .model-info-btn:hover {
      background: #e8f0fe;
      color: var(--accent);
    }
    .model-info-btn svg {
      width: 18px;
      height: 18px;
    }
    .info-grid {
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 8px 14px;
      font-size: 12.5px;
      margin-top: 4px;
    }
    .info-grid dt {
      color: var(--muted);
    }
    .info-grid dd {
      margin: 0;
      color: var(--text);
    }
    .badge {
      font-size: 11px;
      font-weight: 500;
      padding: 3px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #174ea6;
      border: 1px solid #d2e3fc;
      white-space: nowrap;
    }
    .badge.freq {
      background: var(--accent-2-soft);
      color: #7627bb;
      border: 1px solid #e9d2fd;
    }
    .model-desc {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }
    .param-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--border-subtle);
    }
    label {
      display: block;
      font-size: 11.5px;
      font-weight: 500;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    select, input[type="text"], input[type="number"], textarea {
      width: 100%;
      padding: 9px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
      font-size: 13.5px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.14s ease, box-shadow 0.14s ease;
    }
    select:focus, input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.18);
    }
    .prompt-chips {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 14px;
    }
    .prompt-chip {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 9px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
      font-size: 12.5px;
      font-family: inherit;
      cursor: pointer;
      text-align: left;
      transition: all 0.14s ease;
    }
    .prompt-chip:hover {
      border-color: #8ab4f8;
      background: #f8fbff;
    }
    .prompt-chip.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #174ea6;
      font-weight: 500;
    }
    .prompt-cat {
      font-size: 10.5px;
      color: #174ea6;
      background: #ffffff;
      border: 1px solid #d2e3fc;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 500;
    }
    .launch-box {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 13px;
      color: var(--text);
      background: var(--panel-alt);
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .launch-btn {
      width: 100%;
      padding: 13px 20px;
      border-radius: 8px;
      border: none;
      background: var(--accent);
      color: #ffffff;
      font-size: 14.5px;
      font-weight: 500;
      font-family: inherit;
      cursor: pointer;
      letter-spacing: 0.01em;
      box-shadow: 0 1px 2px 0 rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
      transition: background 0.14s ease, box-shadow 0.14s ease;
    }
    .launch-btn:hover {
      background: var(--accent-hover);
      box-shadow: 0 1px 3px 0 rgba(60, 64, 67, 0.3), 0 4px 8px 3px rgba(60, 64, 67, 0.15);
    }
    .launch-btn:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .main-stage-frame {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      margin-top: 12px;
      box-shadow: 0 1px 3px rgba(60, 64, 67, 0.08);
    }
    .main-stage-frame img {
      width: 100%;
      max-height: 460px;
      object-fit: contain;
      background: #ffffff;
      display: block;
      margin: 0 auto;
    }
    .main-stage-bar {
      padding: 10px 14px;
      background: var(--panel-alt);
      border-top: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .main-stage-title {
      font-size: 13px;
      font-weight: 500;
      color: var(--text);
    }
    .main-stage-sub {
      font-size: 11.5px;
      color: var(--muted);
    }
    .thumb-strip-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 16px;
      margin-bottom: 8px;
    }
    .thumb-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 10px;
    }
    .thumb-card {
      background: #ffffff;
      border: 1.5px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.14s ease;
      display: flex;
      flex-direction: column;
    }
    .thumb-card:hover {
      border-color: #8ab4f8;
      transform: translateY(-1px);
      box-shadow: 0 2px 6px rgba(60, 64, 67, 0.12);
    }
    .thumb-card.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
    }
    .thumb-card img {
      width: 100%;
      height: 88px;
      object-fit: cover;
      display: block;
      background: #f8f9fa;
    }
    .thumb-label {
      padding: 6px 8px;
      font-size: 11px;
      font-weight: 500;
      color: var(--text);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 4px;
    }
    .thumb-card.active .thumb-label {
      color: #174ea6;
      font-weight: 700;
    }
    .obj-pills-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--border-subtle);
    }
    .obj-pill {
      font-size: 11.5px;
      padding: 5px 10px;
      border-radius: 999px;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .google-outline-btn {
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--accent);
      font-size: 12px;
      font-weight: 500;
      font-family: inherit;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.14s ease;
    }
    .google-outline-btn:hover {
      background: var(--accent-soft);
      border-color: #d2e3fc;
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(32, 33, 36, 0.48);
      backdrop-filter: blur(2px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
      padding: 20px;
    }
    .modal-backdrop.open {
      display: flex;
    }
    .modal-dialog {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
      max-width: 720px;
      width: 100%;
      max-height: 85vh;
      overflow-y: auto;
      padding: 22px 24px;
      box-shadow: 0 12px 36px rgba(60, 64, 67, 0.28);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-subtle);
      margin-bottom: 14px;
    }
    .modal-title {
      font-size: 16px;
      font-weight: 500;
      color: var(--text);
    }
    .modal-close-btn {
      border: 1px solid var(--border);
      background: var(--panel-alt);
      color: var(--muted);
      width: 30px;
      height: 30px;
      border-radius: 50%;
      font-size: 16px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .modal-close-btn:hover {
      background: #e8eaed;
      color: var(--text);
    }
    .img-frame {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }
    .img-frame img {
      width: 100%;
      display: block;
    }
    .img-caption {
      padding: 8px 12px;
      font-size: 11.5px;
      color: var(--muted);
      background: var(--panel-alt);
      border-top: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      font-weight: 500;
    }
    .meta-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      margin-top: 8px;
    }
    .meta-table th, .meta-table td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--border-subtle);
      text-align: left;
      color: var(--text);
    }
    .meta-table th {
      color: var(--muted);
      font-weight: 500;
      background: var(--panel-alt);
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    .kpi-card {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .kpi-label {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
      font-weight: 500;
    }
    .kpi-val {
      font-size: 16.5px;
      font-weight: 700;
      color: var(--text);
    }
    .cli-bar {
      font-family: 'Roboto Mono', monospace;
      font-size: 12px;
      background: #f1f3f4;
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      color: #174ea6;
      overflow-x: auto;
      margin-bottom: 14px;
    }
    /* Left Slide-Out Saved Runs Drawer */
    .runs-drawer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(32, 33, 36, 0.36);
      z-index: 55;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }
    .runs-drawer-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }
    .runs-drawer {
      position: fixed;
      top: 0;
      left: 0;
      bottom: 0;
      width: 440px;
      max-width: 92vw;
      background: #ffffff;
      border-right: 1px solid var(--border);
      box-shadow: 4px 0 24px rgba(32, 33, 36, 0.18);
      z-index: 60;
      transform: translateX(-102%);
      transition: transform 0.24s cubic-bezier(0.2, 0.8, 0.2, 1);
      display: flex;
      flex-direction: column;
    }
    .runs-drawer.open {
      transform: translateX(0);
    }
    .runs-drawer-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--panel-alt);
    }
    .runs-drawer-list {
      flex: 1;
      overflow-y: auto;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .run-item-card {
      border: 1.5px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      background: #ffffff;
      cursor: pointer;
      transition: all 0.14s ease;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .run-item-card:hover {
      border-color: var(--accent);
      background: #f8fbff;
      transform: translateX(2px);
      box-shadow: 0 2px 6px rgba(26, 115, 232, 0.12);
    }
    .run-item-card.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    /* Dynamic Multi-Camera Grid */
    .multi-cam-grid {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }
    .multi-cam-grid.cols-1 {
      grid-template-columns: minmax(0, 480px);
    }
    .multi-cam-grid.cols-2 {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .multi-cam-grid.cols-3 {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .cam-view-box {
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      background: #ffffff;
      box-shadow: 0 1px 2px rgba(60, 64, 67, 0.08);
      display: flex;
      flex-direction: column;
    }
    .cam-view-box video, .cam-view-box img {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #111827;
      display: block;
    }
    .cam-view-caption {
      padding: 8px 11px;
      background: var(--panel-alt);
      border-top: 1px solid var(--border-subtle);
      font-size: 11.5px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    /* Run Replay & Live Execution Modal */
    .run-modal-dialog {
      background: #ffffff;
      border-radius: 14px;
      border: 1px solid var(--border);
      width: 1220px;
      max-width: 96vw;
      max-height: 92vh;
      overflow-y: auto;
      padding: 22px 26px;
      box-shadow: 0 12px 36px rgba(32, 33, 36, 0.28);
    }
    .run-modal-dialog > .modal-header {
      position: sticky;
      top: -22px;
      z-index: 10;
      margin: -22px -26px 12px;
      padding: 22px 26px 12px;
      background: #ffffff;
    }
    .scene-hero {
      position: relative;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: #111827;
      margin-bottom: 16px;
    }
    .scene-hero img, .scene-hero video {
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      display: block;
      background: #111827;
    }
    .scene-hero-label {
      position: absolute;
      top: 10px;
      left: 12px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.92);
      color: var(--text);
      font-size: 12px;
      font-weight: 500;
      pointer-events: none;
    }
    .live-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: var(--danger);
      font-weight: 600;
      font-size: 11px;
      letter-spacing: 0.04em;
    }
    .live-badge::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--danger);
      animation: live-pulse 1.2s ease-in-out infinite;
    }
    @keyframes live-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.25; }
    }
    .scene-hero-waiting {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #cbd5e1;
      font-size: 13px;
      pointer-events: none;
    }
    .run-meta-banner {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      padding: 12px 14px;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 16px;
      font-size: 12.5px;
    }
  </style>
</head>
<body>
  <!-- LEFT SLIDE-OUT SAVED RUNS DRAWER -->
  <div class="runs-drawer-backdrop" id="runs-drawer-backdrop" onclick="toggleRunsDrawer(false)"></div>
  <aside class="runs-drawer" id="runs-drawer" aria-label="Saved Runs Slider">
    <div class="runs-drawer-header">
      <div>
        <div style="font-size:15.5px;font-weight:600;color:var(--text);display:flex;align-items:center;gap:8px;">
          <span>📂 Saved Simulation Runs</span>
          <span class="badge" id="drawer-runs-count-badge">0 Runs</span>
        </div>
        <div style="font-size:11.5px;color:var(--muted);margin-top:2px;font-family:'Roboto Mono',monospace;">
          Local Directory: data/runs/&lt;timestamp&gt;/
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <button type="button" class="google-outline-btn" id="runs-sort-btn" onclick="toggleRunsSortOrder()" style="padding:4px 9px;font-size:11px;">
          ↓ Newest
        </button>
        <button type="button" class="modal-close-btn" onclick="toggleRunsDrawer(false)" aria-label="Close runs slider">✕</button>
      </div>
    </div>
    <div class="runs-drawer-list" id="runs-drawer-list">
      <div style="padding:24px;text-align:center;color:var(--muted);font-size:13px;">
        No saved runs yet. Click "Launch Live Simulation" to record multi-camera videos and telemetry into <code>data/runs/</code>.
      </div>
    </div>
  </aside>

  <header>
    <div class="brand">
      <button type="button" class="icon-btn" id="header-runs-btn" onclick="toggleRunsDrawer(true)" title="Run history" aria-label="Open run history">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>
        </svg>
        <span class="count-dot" id="header-runs-count">0</span>
      </button>
      <div class="brand-icon" aria-hidden="true">
        <span class="brand-dot b"></span>
        <span class="brand-dot r"></span>
        <span class="brand-dot y"></span>
        <span class="brand-dot g"></span>
      </div>
      <h1>VLA & Spectral Control Studio</h1>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span class="header-status" id="gpu-status-pill" title="Detecting GPU..."><span class="dot cpu"></span><span class="label">Detecting…</span></span>
      <span class="header-status" id="server-status-pill"><span class="dot"></span><span class="label" id="server-status">Ready</span></span>
    </div>
  </header>

  <div class="container">
    <!-- LEFT COLUMN: CONTROLS -->
    <div>
      <!-- STEP 1: MODEL SELECTION (TIME VS FREQUENCY DOMAIN) -->
      <div class="card">
        <div class="step-header">
          <div class="step-title">
            <span class="step-num">1</span>
            <span>Select Control Model</span>
          </div>
          <div class="domain-tabs" role="tablist" style="margin-bottom:0;">
            <button class="domain-tab" id="tab-time" role="tab" onclick="switchDomain('time_domain')">Time domain</button>
            <button class="domain-tab active" id="tab-freq" role="tab" onclick="switchDomain('frequency_domain')">Frequency domain</button>
          </div>
        </div>

        <div class="model-list" id="model-list"></div>

        <div class="param-row" id="spectral-params-row">
          <div>
            <label for="freq-band-input">Normalized Freq Band (Low, High)</label>
            <input type="text" id="freq-band-input" value="0.0,0.35" />
          </div>
          <div>
            <label for="quant-scale-input">Token Quantization Scale</label>
            <input type="number" id="quant-scale-input" value="40" step="5" min="1" max="200" />
          </div>
        </div>
      </div>

      <!-- STEP 2: ROBOT ARRANGEMENT SELECTOR -->
      <div class="card">
        <div class="step-header">
          <div class="step-title">
            <span class="step-num">2</span>
            <span>Select Robotic Arm & Scene Arrangement</span>
          </div>
        </div>
        <label for="arrangement-select">Hardware & Obstacle Arrangement Preset</label>
        <select id="arrangement-select" onchange="onArrangementChange()"></select>
        <p id="arrangement-desc" style="font-size:12.5px;color:var(--muted);margin:10px 0 0;line-height:1.45;"></p>
      </div>

      <!-- STEP 3: PRE-CONFIGURED & CUSTOM PROMPTS -->
      <div class="card">
        <div class="step-header">
          <div class="step-title">
            <span class="step-num">3</span>
            <span>Task Prompt (Pre-Configured or Custom)</span>
          </div>
        </div>
        <label>Pre-Configured Prompts for Selected Setup (Click to Load)</label>
        <div class="prompt-chips" id="prompt-chips"></div>

        <label for="custom-prompt-input">Active / Custom Natural-Language Prompt</label>
        <input type="text" id="custom-prompt-input" value="Move the apple to the target" placeholder="Type any custom instruction, e.g., Place the apple to the right of the Coke can" />
      </div>

      <!-- STEP 4: LAUNCH LIVE SIMULATION -->
      <div class="card">
        <div class="step-header">
          <div class="step-title">
            <span class="step-num">4</span>
            <span>Launch Live Simulation</span>
          </div>
        </div>
        <div class="launch-box">
          <div class="toggle-row">
            <span>Open Live 3D PyBullet GUI Window on Desktop</span>
            <input type="checkbox" id="open-gui-checkbox" checked style="width:18px;height:18px;accent-color:var(--success);" />
          </div>
          <div class="toggle-row">
            <span>GUI Hold Duration After Completion (seconds)</span>
            <input type="number" id="hold-seconds-input" value="6" min="1" max="60" style="width:80px;text-align:center;" />
          </div>
          <button class="launch-btn" id="launch-btn" onclick="launchSimulation()">
            ▶ Launch Live Simulation & Compute Spectral Telemetry
          </button>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN: MAIN 2D TOP-VIEW PANEL, CLICKABLE THUMBNAILS, & TELEMETRY -->
    <div>
      <!-- SCENE & CAMERA ARRANGEMENT PREVIEW -->
      <div class="card">
        <div class="step-header" style="flex-wrap:wrap;gap:8px;">
          <div class="step-title">
            <span>Workspace 2D Top-View & Camera Previews</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span class="badge" id="preview-robot-badge">SO-101 6-DoF</span>
            <button type="button" class="google-outline-btn" onclick="openCameraModal()">
              📷 Camera & Scene Specs
            </button>
          </div>
        </div>

        <!-- MAIN HERO VIEW (DEFAULTS TO 2D TOP-DOWN SCHEMATIC) -->
        <div class="main-stage-frame">
          <img id="main-stage-img" src="" alt="Main Workspace View" />
          <div class="main-stage-bar">
            <div>
              <div class="main-stage-title" id="main-stage-title">2D Top-Down Engineering Schematic (Main View)</div>
              <div class="main-stage-sub" id="main-stage-sub">Shows robot arm base, object coordinates, obstacles, goal marker & camera layout</div>
            </div>
            <button type="button" class="google-outline-btn" onclick="openCameraModal()">
              View Camera Placement & Object XYZ Details
            </button>
          </div>
        </div>

        <!-- CLICKABLE SCREENSHOT THUMBNAILS -->
        <div class="thumb-strip-header">
          <label style="margin:0;">Available Views & Live Camera Screenshots (Click Thumbnail to Inspect)</label>
          <span style="font-size:11.5px;color:var(--muted);" id="thumb-count-label"></span>
        </div>
        <div class="thumb-strip" id="thumb-strip"></div>

        <!-- DYNAMIC CAMERA VIEWS ACCORDING TO NUMBER OF CAMERAS -->
        <div style="margin-top:16px;padding-top:14px;border-top:1px solid var(--border-subtle);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <label style="margin:0;" id="active-cameras-heading">Configured Robot Camera Views (3 Cameras)</label>
            <span class="badge" id="active-cameras-count-badge">3 Active Cameras</span>
          </div>
          <div class="multi-cam-grid cols-3" id="main-cameras-grid"></div>
        </div>

        <!-- COMPACT OBJECT COORDINATE PILLS -->
        <div class="obj-pills-row" id="objects-pills-row"></div>
      </div>

      <!-- LIVE EXECUTION TELEMETRY & STAT NUMBERS -->
      <div class="card" id="telemetry-card">
        <div class="step-header" style="flex-wrap:wrap;gap:8px;">
          <div class="step-title">
            <span>Live Execution Stat Numbers & Multi-Camera Video Replay</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span class="badge" id="run-status-badge">Ready to Launch</span>
            <button type="button" class="google-outline-btn" id="reopen-run-modal-btn" onclick="if(activeRunRecord) openRunModal(activeRunRecord)" style="display:none;">
              🎬 Open Run Modal & Videos
            </button>
          </div>
        </div>

        <div class="cli-bar" id="cli-command-bar">Click "Launch Live Simulation" to execute, record multi-camera videos into data/runs/, and open the live run modal...</div>

        <!-- MAIN PAGE CAMERA VIDEO REPLAY (ACCORDING TO NUMBER OF CAMERAS) -->
        <div id="main-run-videos-section" style="display:none;margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <label style="margin:0;" id="main-run-videos-label">Recorded Multi-Camera Simulation Videos</label>
            <span style="font-size:11.5px;color:var(--muted);font-family:'Roboto Mono',monospace;" id="main-run-folder-label"></span>
          </div>
          <div class="scene-hero" id="main-scene-section" style="display:none;margin-bottom:10px;">
            <video id="main-scene-video" autoplay loop muted playsinline controls></video>
            <div class="scene-hero-label"><span>3D Scene View</span></div>
          </div>
          <div class="multi-cam-grid cols-3" id="main-run-videos-grid"></div>
        </div>

        <!-- STAT NUMBERS GRID -->
        <div class="kpi-grid" id="kpi-grid" style="grid-template-columns:repeat(auto-fit, minmax(145px, 1fr));">
          <div class="kpi-card">
            <div class="kpi-label">Task Success</div>
            <div class="kpi-val" id="kpi-success">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">3D Vertical Lift</div>
            <div class="kpi-val" id="kpi-lift">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Endpoint Error</div>
            <div class="kpi-val" id="kpi-error">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Joint RMS Jerk</div>
            <div class="kpi-val" id="kpi-jerk">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Chunk Boundary Jump</div>
            <div class="kpi-val" id="kpi-jump">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">High-Freq Energy (>5Hz)</div>
            <div class="kpi-val" id="kpi-hfer">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">SPARC Smoothness</div>
            <div class="kpi-val" id="kpi-sparc">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Path Efficiency</div>
            <div class="kpi-val" id="kpi-eff">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Token Compression</div>
            <div class="kpi-val" id="kpi-comp">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Control Latency</div>
            <div class="kpi-val" id="kpi-lat">—</div>
          </div>
        </div>

        <div class="img-frame" id="chart-frame" style="display:none;">
          <img id="diagnostic-chart-img" src="" alt="5-Panel Spectral Diagnostic Chart" />
          <div class="img-caption">
            <span>5-Panel Trajectory & Power Spectral Density (PSD) Analysis</span>
            <span id="chart-meta-caption"></span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- RUN REPLAY & LIVE EXECUTION MODAL POPUP -->
  <div class="modal-backdrop" id="run-modal-backdrop" onclick="if(event.target===this) closeRunModal()">
    <div class="run-modal-dialog" role="dialog" aria-modal="true" aria-labelledby="run-modal-title">
      <div class="modal-header" style="margin-bottom:12px;">
        <div>
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <div class="modal-title" id="run-modal-title">Live Simulation Run & Multi-Camera Video Replay</div>
            <span class="badge" id="run-modal-status-badge" style="background:#e6f4ea;color:#137333;border-color:#ceead6;">Completed</span>
          </div>
          <div style="font-size:12px;color:var(--muted);margin-top:3px;font-family:'Roboto Mono',monospace;" id="run-modal-subtitle">
            data/runs/—
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <button type="button" class="google-outline-btn" onclick="toggleRunsDrawer(true)">
            📂 All Saved Runs
          </button>
          <button type="button" class="modal-close-btn" onclick="closeRunModal()" aria-label="Close run modal">✕</button>
        </div>
      </div>

      <!-- LOADING BANNER WHEN LIVE RUN IS IN PROGRESS -->
      <div id="run-modal-loading" style="display:none;padding:9px 14px;margin-bottom:12px;border-radius:8px;background:#e8f0fe;border:1px solid #d2e3fc;font-size:12.5px;color:#174ea6;">
        Running in PyBullet. Camera videos and stats will appear here when the run finishes.
      </div>

      <!-- 3D SCENE VIEW: LIVE MJPEG STREAM WHILE RUNNING, RECORDED VIDEO AFTERWARDS -->
      <div class="scene-hero" id="modal-scene-section" style="display:none;">
        <img id="modal-scene-live" alt="Live 3D scene view" style="display:none;" />
        <video id="modal-scene-video" style="display:none;" autoplay loop muted playsinline controls></video>
        <div class="scene-hero-waiting" id="modal-scene-waiting" style="display:none;">Starting simulation…</div>
        <div class="scene-hero-label">
          <span>3D Scene View</span>
          <span class="live-badge" id="modal-scene-live-badge" style="display:none;">LIVE</span>
        </div>
      </div>

      <!-- RUN METADATA BANNER (MODEL, PROMPT, TIME, ARM SETUP) -->
      <div class="run-meta-banner" id="run-modal-meta-banner">
        <div>
          <div style="font-size:10.5px;color:var(--muted);text-transform:uppercase;font-weight:600;">Control Model & Version</div>
          <div style="font-weight:600;color:var(--text);margin-top:2px;" id="modal-meta-model">—</div>
          <div style="font-size:11.5px;color:#7627bb;margin-top:1px;" id="modal-meta-version">—</div>
        </div>
        <div>
          <div style="font-size:10.5px;color:var(--muted);text-transform:uppercase;font-weight:600;">Robotic Arm Setup</div>
          <div style="font-weight:600;color:var(--text);margin-top:2px;" id="modal-meta-arm">—</div>
          <div style="font-size:11.5px;color:#1a73e8;margin-top:1px;" id="modal-meta-cameras">—</div>
        </div>
        <div>
          <div style="font-size:10.5px;color:var(--muted);text-transform:uppercase;font-weight:600;">Task Prompt</div>
          <div style="font-weight:600;color:#137333;margin-top:2px;" id="modal-meta-prompt">—</div>
          <div style="font-size:11.5px;color:var(--muted);margin-top:1px;" id="modal-meta-schema">—</div>
        </div>
        <div>
          <div style="font-size:10.5px;color:var(--muted);text-transform:uppercase;font-weight:600;">Timestamp & Local Folder</div>
          <div style="font-weight:600;color:var(--text);margin-top:2px;" id="modal-meta-time">—</div>
          <div style="font-size:11px;font-family:'Roboto Mono',monospace;color:var(--muted);margin-top:1px;" id="modal-meta-folder">—</div>
        </div>
      </div>

      <!-- MULTI-CAMERA VIDEO REPLAY ACCORDING TO NUMBER OF CAMERAS -->
      <div id="run-modal-content-body">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
          <label style="margin:0;font-size:12.5px;color:var(--text);" id="modal-cameras-label">
            📹 Synchronized Multi-Camera Video Replay (3 Cameras)
          </label>
          <div style="display:flex;align-items:center;gap:8px;">
            <button type="button" class="google-outline-btn" onclick="syncModalVideos('play')">▶ Play All Cameras</button>
            <button type="button" class="google-outline-btn" onclick="syncModalVideos('pause')">⏸ Pause All</button>
            <button type="button" class="google-outline-btn" onclick="syncModalVideos('restart')">↺ Replay From Start</button>
          </div>
        </div>
        <div class="multi-cam-grid cols-3" id="modal-cameras-grid" style="margin-bottom:18px;"></div>

        <!-- STAT NUMBERS INSIDE MODAL -->
        <label style="margin:0 0 8px;font-size:12.5px;color:var(--text);">📊 Run Stat Numbers & Spectral Performance Telemetry</label>
        <div class="kpi-grid" id="modal-kpi-grid" style="grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));margin-bottom:18px;">
          <div class="kpi-card">
            <div class="kpi-label">Task Success</div>
            <div class="kpi-val" id="modal-kpi-success">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">3D Vertical Lift</div>
            <div class="kpi-val" id="modal-kpi-lift">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Endpoint Error</div>
            <div class="kpi-val" id="modal-kpi-error">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Joint RMS Jerk</div>
            <div class="kpi-val" id="modal-kpi-jerk">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Chunk Boundary Jump</div>
            <div class="kpi-val" id="modal-kpi-jump">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">High-Freq Energy (>5Hz)</div>
            <div class="kpi-val" id="modal-kpi-hfer">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">SPARC Smoothness</div>
            <div class="kpi-val" id="modal-kpi-sparc">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Path Efficiency</div>
            <div class="kpi-val" id="modal-kpi-eff">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Token Compression</div>
            <div class="kpi-val" id="modal-kpi-comp">—</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Control Latency</div>
            <div class="kpi-val" id="modal-kpi-lat">—</div>
          </div>
        </div>

        <!-- 5-PANEL SPECTRAL DIAGNOSTIC CHART -->
        <div class="img-frame" id="modal-chart-frame" style="margin-top:8px;">
          <img id="modal-chart-img" src="" alt="5-Panel Spectral Diagnostic Chart" />
          <div class="img-caption">
            <span>5-Panel Trajectory & Power Spectral Density (PSD) Analysis</span>
            <span id="modal-chart-caption"></span>
          </div>
        </div>
      </div>

      <div style="margin-top:18px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div style="font-size:11.5px;font-family:'Roboto Mono',monospace;color:var(--muted);" id="modal-cli-text"></div>
        <button type="button" class="google-outline-btn" onclick="closeRunModal()">Close Modal</button>
      </div>
    </div>
  </div>

  <!-- MODEL INFO MODAL -->
  <div class="modal-backdrop" id="model-info-backdrop" onclick="if(event.target===this) closeModelInfo()">
    <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="model-info-title" style="max-width:560px;">
      <div class="modal-header">
        <div>
          <div class="modal-title" id="model-info-title">Model</div>
          <div style="font-size:12px;color:var(--muted);margin-top:2px;" id="model-info-subtitle"></div>
        </div>
        <button type="button" class="modal-close-btn" onclick="closeModelInfo()" aria-label="Close model info">✕</button>
      </div>
      <p style="font-size:13px;line-height:1.55;color:var(--text);margin:0 0 14px;" id="model-info-desc"></p>
      <dl class="info-grid" id="model-info-grid"></dl>
      <div style="margin-top:18px;display:flex;justify-content:flex-end;gap:8px;">
        <button type="button" class="google-outline-btn" onclick="closeModelInfo()">Close</button>
        <button type="button" class="google-outline-btn" id="model-info-select-btn" style="background:var(--accent);color:#ffffff;border-color:var(--accent);">Use this model</button>
      </div>
    </div>
  </div>

  <!-- CAMERA PLACEMENT & OBJECT COORDINATES MODAL POPUP -->
  <div class="modal-backdrop" id="camera-modal-backdrop" onclick="if(event.target===this) closeCameraModal()">
    <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="camera-modal-title">
      <div class="modal-header">
        <div>
          <div class="modal-title" id="camera-modal-title">Camera Placement, FOV & Object Coordinates</div>
          <div style="font-size:12px;color:var(--muted);margin-top:2px;" id="camera-modal-subtitle"></div>
        </div>
        <button type="button" class="modal-close-btn" onclick="closeCameraModal()" aria-label="Close modal">✕</button>
      </div>

      <label>Camera Placement & Role Specification</label>
      <table class="meta-table" id="cameras-table">
        <thead><tr><th>Camera</th><th>Placement / FOV</th><th>Role</th></tr></thead>
        <tbody></tbody>
      </table>

      <label style="margin-top:18px;">Initial Object Coordinates (World XYZ [m])</label>
      <table class="meta-table" id="objects-table">
        <thead><tr><th>Object</th><th>Position (X, Y, Z)</th></tr></thead>
        <tbody></tbody>
      </table>

      <div style="margin-top:18px;display:flex;justify-content:flex-end;">
        <button type="button" class="google-outline-btn" onclick="closeCameraModal()">Close</button>
      </div>
    </div>
  </div>

  <script>
    let catalogData = null;
    let currentDomain = 'frequency_domain';
    let selectedModel = null;
    let currentArrangement = null;
    let savedRunsList = [];
    let runsSortNewest = true;
    let activeRunRecord = null;

    function setServerStatus(text, busy) {
      document.getElementById('server-status').textContent = text;
      document.querySelector('#server-status-pill .dot').className = busy ? 'dot busy' : 'dot';
    }

    function toggleRunsDrawer(open) {
      const drawer = document.getElementById('runs-drawer');
      const backdrop = document.getElementById('runs-drawer-backdrop');
      if (open === undefined) {
        open = !drawer.classList.contains('open');
      }
      drawer.classList.toggle('open', open);
      backdrop.classList.toggle('open', open);
    }

    function toggleRunsSortOrder() {
      runsSortNewest = !runsSortNewest;
      document.getElementById('runs-sort-btn').textContent = runsSortNewest ? '↓ Newest' : '↑ Oldest';
      renderSavedRunsList();
    }

    async function refreshSavedRuns() {
      try {
        const res = await fetch('/api/runs');
        const data = await res.json();
        savedRunsList = data.runs || [];
        const countEl = document.getElementById('header-runs-count');
        countEl.textContent = savedRunsList.length;
        countEl.style.display = savedRunsList.length ? 'inline-block' : 'none';
        document.getElementById('drawer-runs-count-badge').textContent = `${savedRunsList.length} Run${savedRunsList.length === 1 ? '' : 's'}`;
        renderSavedRunsList();
      } catch (e) {
        console.error('Failed to load saved runs:', e);
      }
    }

    function renderSavedRunsList() {
      const listEl = document.getElementById('runs-drawer-list');
      if (!savedRunsList || savedRunsList.length === 0) {
        listEl.innerHTML = `
          <div style="padding:24px;text-align:center;color:var(--muted);font-size:13px;">
            No saved runs yet. Click "Launch Live Simulation" to record multi-camera videos and telemetry into <code>data/runs/</code>.
          </div>
        `;
        return;
      }
      const ordered = [...savedRunsList];
      if (!runsSortNewest) {
        ordered.reverse();
      }
      listEl.innerHTML = '';
      ordered.forEach((run) => {
        const m = run.metrics || {};
        const pass = Boolean(m.task_success);
        const card = document.createElement('div');
        card.className = 'run-item-card' + (activeRunRecord && activeRunRecord.run_id === run.run_id ? ' active' : '');
        card.onclick = () => {
          toggleRunsDrawer(false);
          openRunModal(run);
        };
        card.innerHTML = `
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
            <span style="font-size:11.5px;font-family:'Roboto Mono',monospace;color:var(--muted);font-weight:600;">
              🕒 ${run.timestamp || run.run_id}
            </span>
            <span class="badge" style="background:${pass ? '#e6f4ea' : '#fce8e6'};color:${pass ? '#137333' : '#d93025'};border-color:${pass ? '#ceead6' : '#fad2cf'};">
              ${pass ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div style="font-size:13px;font-weight:600;color:var(--text);line-height:1.35;">
            ${run.model_display_name || run.model}
          </div>
          <div style="display:flex;gap:5px;flex-wrap:wrap;align-items:center;">
            <span class="badge" style="background:#f3e8fd;color:#7627bb;border-color:#e9d2fd;font-size:10.5px;">
              ${run.model_version || run.spectrum_schema}
            </span>
            <span class="badge" style="font-size:10.5px;">
              ${run.camera_count || (run.cameras ? run.cameras.length : 1)} Cam${(run.camera_count || 1) > 1 ? 's' : ''}
            </span>
          </div>
          <div style="font-size:12px;color:#174ea6;font-weight:500;">
            🦾 ${run.arrangement_title || run.robot.toUpperCase()}
          </div>
          <div style="font-size:12px;color:#137333;background:#f8f9fa;padding:5px 8px;border-radius:6px;border:1px solid var(--border-subtle);">
            💬 "${run.prompt}"
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--muted);padding-top:2px;">
            <span>Folder: <code>${run.run_id}</code></span>
            <span>Lift: +${(m.max_vertical_lift_cm || 0).toFixed(1)}cm • Jerk: ${(m.joint_rms_jerk_rad_s3 || 0).toFixed(1)}</span>
          </div>
        `;
        listEl.appendChild(card);
      });
    }

    function populateKpiStats(prefix, m) {
      const pass = Boolean(m.task_success);
      const sucEl = document.getElementById(`${prefix}-success`);
      if (sucEl) {
        sucEl.textContent = pass ? 'PASS (100%)' : 'FAIL';
        sucEl.style.color = pass ? '#1e8e3e' : '#d93025';
      }
      const setVal = (suffix, text) => {
        const el = document.getElementById(`${prefix}-${suffix}`);
        if (el) el.textContent = text;
      };
      setVal('lift', '+' + (m.max_vertical_lift_cm || 0).toFixed(2) + ' cm');
      setVal('error', (m.endpoint_error_m || 0).toFixed(4) + ' m');
      setVal('jerk', (m.joint_rms_jerk_rad_s3 || 0).toFixed(1) + ' rad/s³');
      setVal('jump', (m.chunk_boundary_jump_rad_s || 0).toFixed(3) + ' rad/s');
      setVal('hfer', ((m.high_freq_energy_ratio_hfer || 0) * 100).toFixed(2) + '%');
      setVal('sparc', (m.sparc_smoothness || 0).toFixed(2));
      setVal('eff', ((m.path_efficiency_ratio || 0) * 100).toFixed(1) + '%');
      setVal('comp', ((m.token_compression_ratio || 0) * 100).toFixed(1) + '%');
      setVal('lat', (m.total_control_latency_ms || 0).toFixed(2) + ' ms');
    }

    function buildVideoGridHTML(cameras, videoClass) {
      return (cameras || []).map((cam) => `
        <div class="cam-view-box">
          <video class="${videoClass}" src="${cam.video_url}" autoplay loop muted playsinline controls></video>
          <div class="cam-view-caption">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <strong style="color:var(--text);">${cam.name}</strong>
              <span class="badge" style="font-size:10px;padding:1px 6px;">FOV ${cam.fov_deg || 60}°</span>
            </div>
            <span style="color:var(--muted);">${cam.role || ''}</span>
          </div>
        </div>
      `).join('');
    }

    function startLiveSceneView() {
      const section = document.getElementById('modal-scene-section');
      const img = document.getElementById('modal-scene-live');
      const vid = document.getElementById('modal-scene-video');
      const waiting = document.getElementById('modal-scene-waiting');
      vid.pause();
      vid.removeAttribute('src');
      vid.style.display = 'none';
      section.style.display = 'block';
      waiting.style.display = 'flex';
      img.style.display = 'block';
      img.onload = () => { waiting.style.display = 'none'; };
      img.src = `/api/live/stream?t=${Date.now()}`;
      document.getElementById('modal-scene-live-badge').style.display = 'inline-flex';
    }

    function stopLiveSceneView() {
      const img = document.getElementById('modal-scene-live');
      img.onload = null;
      img.removeAttribute('src');
      img.style.display = 'none';
      document.getElementById('modal-scene-waiting').style.display = 'none';
      document.getElementById('modal-scene-live-badge').style.display = 'none';
    }

    function showRecordedSceneView(run) {
      stopLiveSceneView();
      const section = document.getElementById('modal-scene-section');
      const vid = document.getElementById('modal-scene-video');
      if (!run.scene_video) {
        section.style.display = 'none';
        vid.removeAttribute('src');
        return;
      }
      section.style.display = 'block';
      vid.style.display = 'block';
      vid.src = run.scene_video.video_url;
      vid.play().catch(() => {});
    }

    function syncModalVideos(action) {
      const vids = document.querySelectorAll('#modal-scene-video, #modal-cameras-grid video');
      vids.forEach((v) => {
        if (action === 'play') v.play();
        else if (action === 'pause') v.pause();
        else if (action === 'restart') {
          v.currentTime = 0;
          v.play();
        }
      });
    }

    function openRunModalLoading(payload, modelObj, arrangementObj) {
      document.getElementById('run-modal-backdrop').classList.add('open');
      document.getElementById('run-modal-loading').style.display = 'block';
      document.getElementById('run-modal-content-body').style.opacity = '0.35';
      startLiveSceneView();
      document.getElementById('run-modal-title').textContent = 'Executing Live Simulation Run...';
      const statusBadge = document.getElementById('run-modal-status-badge');
      statusBadge.textContent = 'Running PyBullet...';
      statusBadge.style.background = '#fef7e0';
      statusBadge.style.color = '#b06000';
      document.getElementById('run-modal-subtitle').textContent = 'Recording multi-camera videos to data/runs/<timestamp>/...';

      document.getElementById('modal-meta-model').textContent = modelObj ? modelObj.name : payload.model;
      document.getElementById('modal-meta-version').textContent = modelObj ? `${modelObj.version} (${modelObj.checkpoint})` : payload.spectrum_schema;
      document.getElementById('modal-meta-arm').textContent = arrangementObj ? arrangementObj.title : payload.arrangement_id;
      const camCount = arrangementObj && arrangementObj.cameras ? arrangementObj.cameras.length : 1;
      document.getElementById('modal-meta-cameras').textContent = `${camCount} Camera Stream${camCount > 1 ? 's' : ''} Active`;
      document.getElementById('modal-meta-prompt').textContent = `"${payload.prompt}"`;
      document.getElementById('modal-meta-schema').textContent = `Schema: ${payload.spectrum_schema} | Band: [${payload.freq_band}]`;
      document.getElementById('modal-meta-time').textContent = new Date().toLocaleString();
      document.getElementById('modal-meta-folder').textContent = 'data/runs/ (creating timestamp folder...)';
    }

    function openRunModal(run) {
      if (!run) return;
      activeRunRecord = run;
      renderSavedRunsList();

      document.getElementById('run-modal-backdrop').classList.add('open');
      document.getElementById('run-modal-loading').style.display = 'none';
      document.getElementById('run-modal-content-body').style.opacity = '1.0';
      showRecordedSceneView(run);

      const m = run.metrics || {};
      const pass = Boolean(m.task_success);
      document.getElementById('run-modal-title').textContent = `Simulation Run Replay — ${run.run_id}`;
      const statusBadge = document.getElementById('run-modal-status-badge');
      statusBadge.textContent = pass ? 'PASS (100%)' : 'FAIL';
      statusBadge.style.background = pass ? '#e6f4ea' : '#fce8e6';
      statusBadge.style.color = pass ? '#137333' : '#d93025';
      document.getElementById('run-modal-subtitle').textContent = `Saved in ${run.run_dir || ('data/runs/' + run.run_id)} • Recorded at ${run.timestamp}`;

      document.getElementById('modal-meta-model').textContent = run.model_display_name || run.model;
      document.getElementById('modal-meta-version').textContent = `${run.model_version || ''} (${run.checkpoint || ''})`;
      document.getElementById('modal-meta-arm').textContent = run.arrangement_title || run.robot.toUpperCase();
      const cams = run.cameras || [];
      document.getElementById('modal-meta-cameras').textContent = `${cams.length} Camera Video${cams.length === 1 ? '' : 's'} Recorded`;
      document.getElementById('modal-meta-prompt').textContent = `"${run.prompt}"`;
      document.getElementById('modal-meta-schema').textContent = `Schema: ${run.spectrum_schema} | Band: [${(run.freq_band || []).join(', ')}] | Quant: ${run.quant_scale}`;
      document.getElementById('modal-meta-time').textContent = run.timestamp || run.run_id;
      document.getElementById('modal-meta-folder').textContent = run.run_dir || `data/runs/${run.run_id}`;

      // Render camera video players according to number of cameras
      const camCount = Math.max(1, Math.min(3, cams.length));
      document.getElementById('modal-cameras-label').textContent =
        `📹 Synchronized Multi-Camera Video Replay (${cams.length} Camera${cams.length === 1 ? '' : 's'})`;
      const modalGrid = document.getElementById('modal-cameras-grid');
      modalGrid.className = `multi-cam-grid cols-${camCount}`;
      modalGrid.innerHTML = buildVideoGridHTML(cams, 'modal-cam-video');

      // Populate modal stat numbers
      populateKpiStats('modal-kpi', m);

      // Populate modal 5-panel chart
      document.getElementById('modal-chart-img').src = run.chart_url;
      document.getElementById('modal-chart-caption').textContent =
        `Run ID: ${run.run_id} | Schema: ${run.spectrum_schema} | SPARC: ${(m.sparc_smoothness || 0).toFixed(2)}`;
      document.getElementById('modal-cli-text').textContent = '$ ' + (run.cli_command || '');

      // Also update main page telemetry & camera videos
      updateMainPageWithRun(run);
    }

    function closeRunModal() {
      document.getElementById('run-modal-backdrop').classList.remove('open');
      stopLiveSceneView();
      document.querySelectorAll('#modal-scene-video, #modal-cameras-grid video').forEach((v) => v.pause());
    }

    function updateMainPageWithRun(run) {
      if (!run) return;
      const m = run.metrics || {};
      document.getElementById('cli-command-bar').textContent = '$ ' + (run.cli_command || '');
      document.getElementById('reopen-run-modal-btn').style.display = 'inline-flex';
      populateKpiStats('kpi', m);

      const cams = run.cameras || [];
      const camCount = Math.max(1, Math.min(3, cams.length));
      const mainVidSec = document.getElementById('main-run-videos-section');
      mainVidSec.style.display = 'block';
      document.getElementById('main-run-videos-label').textContent =
        `Recorded Multi-Camera Simulation Videos (${cams.length} Camera${cams.length === 1 ? '' : 's'})`;
      document.getElementById('main-run-folder-label').textContent = run.run_dir || `data/runs/${run.run_id}`;
      const mainVidGrid = document.getElementById('main-run-videos-grid');
      mainVidGrid.className = `multi-cam-grid cols-${camCount}`;
      mainVidGrid.innerHTML = buildVideoGridHTML(cams, 'main-cam-video');

      const mainScene = document.getElementById('main-scene-section');
      const mainSceneVid = document.getElementById('main-scene-video');
      if (run.scene_video) {
        mainScene.style.display = 'block';
        mainSceneVid.src = run.scene_video.video_url;
        mainSceneVid.play().catch(() => {});
      } else {
        mainScene.style.display = 'none';
        mainSceneVid.removeAttribute('src');
      }

      const chartFrame = document.getElementById('chart-frame');
      chartFrame.style.display = 'block';
      document.getElementById('diagnostic-chart-img').src = run.chart_url;
      document.getElementById('chart-meta-caption').textContent =
        `Run: ${run.run_id} | Schema: ${run.spectrum_schema} | SPARC: ${(m.sparc_smoothness || 0).toFixed(2)}`;
      document.getElementById('run-status-badge').textContent =
        `Saved: data/runs/${run.run_id}`;
    }

    async function initStudio() {
      const res = await fetch('/api/catalog');
      catalogData = await res.json();

      if (catalogData.gpu_status) {
        const g = catalogData.gpu_status;
        const gpuPill = document.getElementById('gpu-status-pill');
        const dot = gpuPill.querySelector('.dot');
        const label = gpuPill.querySelector('.label');
        if (g.has_nvidia_gpu) {
          label.textContent = g.pybullet_egl_gpu_rendering ? 'GPU' : 'GPU · CPU render';
          dot.className = 'dot';
          gpuPill.title =
            `${g.gpu_name} (${g.vram_total}) · driver ${g.driver_version}\n` +
            `PyBullet: ${g.pybullet_egl_gpu_rendering ? 'EGL hardware OpenGL' : 'TinyRenderer (CPU)'}`;
        } else {
          label.textContent = 'CPU';
          dot.className = 'dot cpu';
          gpuPill.title = 'No NVIDIA GPU detected · PyBullet TinyRenderer (CPU)';
        }
      }

      const arrSelect = document.getElementById('arrangement-select');
      arrSelect.innerHTML = '';
      for (const arr of catalogData.arrangements) {
        const opt = document.createElement('option');
        opt.value = arr.id;
        opt.textContent = arr.title;
        arrSelect.appendChild(opt);
      }

      switchDomain('frequency_domain');
      await onArrangementChange();
      await refreshSavedRuns();
    }

    function switchDomain(domain) {
      currentDomain = domain;
      document.getElementById('tab-time').classList.toggle('active', domain === 'time_domain');
      document.getElementById('tab-freq').classList.toggle('active', domain === 'frequency_domain');

      const models = catalogData.models[domain] || [];
      const listEl = document.getElementById('model-list');
      listEl.innerHTML = '';

      models.forEach((m, idx) => {
        const shortName = m.name.replace(/\s*\(.*\)\s*$/, '');
        const div = document.createElement('div');
        div.className = 'model-item' + (idx === 0 ? ' selected' : '');
        div.title = m.name;
        div.onclick = () => selectModelItem(m, div);
        div.innerHTML = `
          <span class="model-radio" aria-hidden="true"></span>
          <div class="model-item-body">
            <span class="model-name">${shortName}</span>
            <span class="badge ${domain === 'frequency_domain' ? 'freq' : ''}" style="font-size:10.5px;padding:2px 7px;">${m.badge}</span>
          </div>
          <button type="button" class="model-info-btn" title="Model details" aria-label="Show details for ${shortName}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
              <circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>
            </svg>
          </button>
        `;
        div.querySelector('.model-info-btn').onclick = (ev) => {
          ev.stopPropagation();
          openModelInfo(m, div);
        };
        listEl.appendChild(div);
        if (idx === 0) {
          selectModelItem(m, div);
        }
      });

      document.getElementById('spectral-params-row').style.opacity =
        domain === 'frequency_domain' ? '1.0' : '0.45';
    }

    function selectModelItem(modelObj, elem) {
      selectedModel = modelObj;
      document.querySelectorAll('.model-item').forEach(el => el.classList.remove('selected'));
      elem.classList.add('selected');
      document.getElementById('freq-band-input').value = modelObj.freq_band || '0.0,0.35';
    }

    function openModelInfo(modelObj, elem) {
      const isFreq = currentDomain === 'frequency_domain';
      document.getElementById('model-info-title').textContent = modelObj.name;
      document.getElementById('model-info-subtitle').textContent =
        isFreq ? 'Frequency-domain model' : 'Time-domain model';
      document.getElementById('model-info-desc').textContent = modelObj.description;
      const rows = [
        ['Version', modelObj.version],
        ['Checkpoint', `<code style="font-family:'Roboto Mono',monospace;font-size:12px;">${modelObj.checkpoint}</code>`],
        ['Policy', modelObj.model],
        ['Spectrum schema', modelObj.spectrum_schema],
        ['FAST tokenization', modelObj.fast_tokenization ? 'Enabled' : 'Disabled'],
        ['Default freq band', modelObj.freq_band],
      ];
      document.getElementById('model-info-grid').innerHTML =
        rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
      document.getElementById('model-info-select-btn').onclick = () => {
        selectModelItem(modelObj, elem);
        closeModelInfo();
      };
      document.getElementById('model-info-backdrop').classList.add('open');
    }

    function closeModelInfo() {
      document.getElementById('model-info-backdrop').classList.remove('open');
    }

    function openCameraModal() {
      document.getElementById('camera-modal-backdrop').classList.add('open');
    }

    function closeCameraModal() {
      document.getElementById('camera-modal-backdrop').classList.remove('open');
    }

    function selectStageView(item, cardElem) {
      document.querySelectorAll('.thumb-card').forEach(el => el.classList.remove('active'));
      if (cardElem) cardElem.classList.add('active');
      document.getElementById('main-stage-img').src = item.src;
      document.getElementById('main-stage-title').textContent = item.title;
      document.getElementById('main-stage-sub').textContent = item.subtitle;
    }

    async function onArrangementChange() {
      const arrId = document.getElementById('arrangement-select').value;
      setServerStatus('Rendering…', true);
      const res = await fetch(`/api/preview?arrangement_id=${encodeURIComponent(arrId)}`);
      const data = await res.json();
      currentArrangement = data;

      document.getElementById('arrangement-desc').textContent = data.description;
      document.getElementById('preview-robot-badge').textContent = data.badge;
      document.getElementById('camera-modal-subtitle').textContent = `${data.title} — ${data.badge}`;

      // Build clickable view items with 2D Top-Down Schematic as the primary default view
      const stageItems = [
        {
          id: 'schematic_2d',
          shortLabel: '2D Top-View (Main)',
          badgeText: '2D Map',
          src: data.schematic_image,
          title: '2D Top-Down Engineering Schematic (Main View)',
          subtitle: 'Shows robot arm base, object XYZ coordinates, obstacles, goal marker & camera layout',
        },
      ];

      for (const cam of data.cameras) {
        const b64 = data.camera_images[cam.id];
        if (!b64) continue;
        const eyeStr = Array.isArray(cam.eye) ? `(${cam.eye.join(', ')})m` : cam.eye;
        stageItems.push({
          id: cam.id,
          shortLabel: cam.name,
          badgeText: `FOV ${cam.fov_deg}°`,
          src: b64,
          title: `${cam.name} (Live PyBullet Render)`,
          subtitle: `Placement: ${eyeStr} | FOV ${cam.fov_deg}° — ${cam.role}`,
        });
      }

      document.getElementById('thumb-count-label').textContent =
        `${stageItems.length} views available (2D Top-View + ${stageItems.length - 1} camera screenshots)`;

      const thumbStrip = document.getElementById('thumb-strip');
      thumbStrip.innerHTML = '';
      stageItems.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = 'thumb-card' + (idx === 0 ? ' active' : '');
        card.onclick = () => selectStageView(item, card);
        card.innerHTML = `
          <img src="${item.src}" alt="${item.shortLabel}" />
          <div class="thumb-label">
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.shortLabel}</span>
            <span style="font-size:10px;color:var(--muted);">${item.badgeText}</span>
          </div>
        `;
        thumbStrip.appendChild(card);
        if (idx === 0) {
          selectStageView(item, card);
        }
      });

      // Render dynamic camera views grid according to number of cameras configured for this robot
      const camCount = data.cameras.length;
      document.getElementById('active-cameras-heading').textContent =
        `Configured Robot Camera Views (${camCount} Camera${camCount === 1 ? '' : 's'})`;
      document.getElementById('active-cameras-count-badge').textContent =
        `${camCount} Active Camera${camCount === 1 ? '' : 's'}`;
      const mainCamGrid = document.getElementById('main-cameras-grid');
      mainCamGrid.className = `multi-cam-grid cols-${Math.max(1, Math.min(3, camCount))}`;
      mainCamGrid.innerHTML = data.cameras.map((cam) => {
        const b64 = data.camera_images[cam.id] || '';
        return `
          <div class="cam-view-box">
            <img src="${b64}" alt="${cam.name}" />
            <div class="cam-view-caption">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <strong style="color:var(--text);">${cam.name}</strong>
                <span class="badge" style="font-size:10px;padding:1px 6px;">FOV ${cam.fov_deg}°</span>
              </div>
              <span style="color:var(--muted);">${cam.role}</span>
            </div>
          </div>
        `;
      }).join('');

      // Populate compact object coordinate pills under thumbnails
      const pillsRow = document.getElementById('objects-pills-row');
      pillsRow.innerHTML = '';
      for (const [name, xyz] of Object.entries(data.objects)) {
        const pill = document.createElement('div');
        pill.className = 'obj-pill';
        pill.innerHTML = `<strong style="color:#1a73e8;">${name}:</strong> <span>(${xyz[0].toFixed(2)}, ${xyz[1].toFixed(2)}, ${xyz[2].toFixed(2)}) m</span>`;
        pillsRow.appendChild(pill);
      }

      // Populate modal tables
      const objBody = document.querySelector('#objects-table tbody');
      objBody.innerHTML = '';
      for (const [name, xyz] of Object.entries(data.objects)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td><strong>${name}</strong></td><td>(${xyz[0].toFixed(3)}, ${xyz[1].toFixed(3)}, ${xyz[2].toFixed(3)}) m</td>`;
        objBody.appendChild(tr);
      }

      const camBody = document.querySelector('#cameras-table tbody');
      camBody.innerHTML = '';
      for (const cam of data.cameras) {
        const eyeStr = Array.isArray(cam.eye) ? `(${cam.eye.join(', ')})m` : cam.eye;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td><strong>${cam.name}</strong></td><td>${eyeStr} | FOV ${cam.fov_deg}°</td><td>${cam.role}</td>`;
        camBody.appendChild(tr);
      }

      // Populate pre-configured prompts
      const chipsEl = document.getElementById('prompt-chips');
      chipsEl.innerHTML = '';
      data.prompts.forEach((p, idx) => {
        const btn = document.createElement('button');
        btn.className = 'prompt-chip' + (idx === 0 ? ' active' : '');
        btn.onclick = () => {
          document.querySelectorAll('.prompt-chip').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          document.getElementById('custom-prompt-input').value = p.prompt;
        };
        btn.innerHTML = `<span>${p.prompt}</span><span class="prompt-cat">${p.category}</span>`;
        chipsEl.appendChild(btn);
        if (idx === 0) {
          document.getElementById('custom-prompt-input').value = p.prompt;
        }
      });

      setServerStatus('Ready', false);
    }

    async function launchSimulation() {
      if (!selectedModel || !currentArrangement) return;
      const btn = document.getElementById('launch-btn');
      btn.disabled = true;
      btn.textContent = '⏳ Launching Live PyBullet Simulation & Saving Videos...';
      document.getElementById('run-status-badge').textContent = 'Running Simulation...';
      setServerStatus('Running…', true);

      const payload = {
        arrangement_id: currentArrangement.id,
        model_id: selectedModel.id,
        model: selectedModel.model,
        fast_tokenization: selectedModel.fast_tokenization,
        spectrum_schema: selectedModel.spectrum_schema,
        freq_band: document.getElementById('freq-band-input').value,
        quant_scale: parseFloat(document.getElementById('quant-scale-input').value || '40'),
        prompt: document.getElementById('custom-prompt-input').value,
        open_gui: document.getElementById('open-gui-checkbox').checked,
        hold_seconds: parseFloat(document.getElementById('hold-seconds-input').value || '6'),
      };

      // Immediately open the Run Modal so the user can watch the live run & results
      openRunModalLoading(payload, selectedModel, currentArrangement);

      try {
        const res = await fetch('/api/launch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.error) {
          closeRunModal();
          alert('Simulation Error: ' + data.error);
          return;
        }

        await refreshSavedRuns();
        if (data.run) {
          openRunModal(data.run);
        }
      } finally {
        btn.disabled = false;
        btn.textContent = '▶ Launch Live Simulation & Compute Spectral Telemetry';
        setServerStatus('Ready', false);
      }
    }

    document.addEventListener('keydown', (ev) => {
      if (ev.key !== 'Escape') return;
      closeModelInfo();
      closeCameraModal();
      closeRunModal();
      toggleRunsDrawer(false);
    });

    initStudio();
  </script>
</body>
</html>
"""


class SmoothOperatorWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
            return
        file_size = file_path.stat().st_size
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                byte_range = range_header.split("=", 1)[1].split(",")
                start_str, end_str = byte_range[0].split("-", 1)
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                end = min(end, file_size - 1)
                length = max(0, end - start + 1)
                with file_path.open("rb") as f:
                    f.seek(start)
                    chunk = f.read(length)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            except Exception:
                pass
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream_live_frames(self) -> None:
        boundary = "liveframe"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        last_seq = -1
        seen_active = False
        idle_since = time.monotonic()
        try:
            while True:
                seq, frame, active = _LIVE_BROKER.wait_next(last_seq, timeout=1.0)
                seen_active = seen_active or active
                if not seen_active:
                    last_seq = seq
                    if time.monotonic() - idle_since > 60.0:
                        break
                    continue
                if seq != last_seq and frame is not None:
                    last_seq = seq
                    idle_since = time.monotonic()
                    self.wfile.write(
                        f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                else:
                    last_seq = seq
                if not active and frame is not None:
                    break
                if time.monotonic() - idle_since > 60.0:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/catalog":
            arrangements_summary = [
                {
                    "id": k,
                    "title": v["title"],
                    "robot": v["robot"],
                    "seed": v["seed"],
                    "badge": v["badge"],
                    "camera_count": len(v["cameras"]),
                }
                for k, v in ARRANGEMENT_CATALOG.items()
            ]
            self._send_json(
                {
                    "models": MODEL_CATALOG,
                    "arrangements": arrangements_summary,
                    "gpu_status": get_gpu_hardware_status(),
                }
            )
            return

        if parsed.path == "/api/runs":
            self._send_json({"runs": list_saved_runs()})
            return

        if parsed.path == "/api/live/stream":
            self._stream_live_frames()
            return

        if parsed.path.startswith("/api/runs/"):
            rel = parsed.path[len("/api/runs/") :].strip("/")
            parts = [p for p in rel.split("/") if p and ".." not in p]
            if len(parts) == 1:
                meta_path = RUNS_DIR / parts[0] / "metadata.json"
                if meta_path.exists():
                    self._send_json(json.loads(meta_path.read_text(encoding="utf-8")))
                else:
                    self._send_json({"error": "Run not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if len(parts) == 2:
                target_file = RUNS_DIR / parts[0] / parts[1]
                suffix = target_file.suffix.lower()
                ctype = (
                    "video/mp4"
                    if suffix == ".mp4"
                    else "image/png"
                    if suffix == ".png"
                    else "image/jpeg"
                    if suffix in (".jpg", ".jpeg")
                    else "application/octet-stream"
                )
                self._send_file(target_file, ctype)
                return

        if parsed.path == "/api/preview":
            qs = parse_qs(parsed.query)
            arr_id = qs.get("arrangement_id", ["so101_seed_9"])[0]
            try:
                preview = build_arrangement_preview(arr_id)
                self._send_json(preview)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/launch":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                params = json.loads(raw)
                result = execute_web_launch(params)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


class SmoothOperatorHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = SmoothOperatorHTTPServer((args.host, args.port), SmoothOperatorWebHandler)
    print(
        f"Smooth Operator Web Studio listening on http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

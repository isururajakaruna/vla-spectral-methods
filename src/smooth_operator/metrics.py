"""Comprehensive time-domain, spectral, and task-level manipulation metrics.

Implements the Scope 2 evaluation matrix:
- Task & Physical Safety: task_success, endpoint_error_m, collision_count,
  grasp_slip_or_drop, path_efficiency_ratio
- Time-Domain Unsmoothness: joint_rms_jerk_rad_s3, joint_max_jerk_rad_s3,
  ee_rms_jerk_m_s3, object_max_jerk_m_s3, object_rms_jerk_m_s3,
  chunk_boundary_jump_rad_s, dimensionless_jerk_ldlj
- Frequency-Domain Smoothness: sparc_smoothness (Spectral Arc Length),
  high_freq_energy_ratio_hfer, spectral_centroid_hz, token_compression_ratio
- Latency: mean_model_inference_ms, mean_spectrum_decode_ms,
  total_control_latency_ms, episode_wall_time_s
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


def compute_sparc(
    speed_profile: Sequence[float],
    hz: float = 30.0,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float:
    """Compute Spectral Arc Length (SPARC) smoothness metric on a speed profile.

    Returns a negative dimensionless value; values closer to 0.0 indicate
    smoother trajectories (fewer high-frequency sub-movements).
    """
    v = np.asarray(speed_profile, dtype=float).reshape(-1)
    if len(v) < 4 or float(np.max(np.abs(v))) < 1e-9:
        return 0.0

    nfft = int(2 ** (math.ceil(math.log2(len(v))) + padlevel))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(hz))
    spectrum = np.abs(np.fft.rfft(v, n=nfft))
    dc = float(spectrum[0]) if spectrum[0] > 1e-12 else float(np.max(spectrum))
    if dc <= 1e-12:
        return 0.0
    mf = spectrum / dc

    fc_mask = freqs <= float(fc)
    freqs_fc = freqs[fc_mask]
    mf_fc = mf[fc_mask]

    above_th = np.where(mf_fc >= amp_th)[0]
    if len(above_th) == 0:
        return 0.0
    fc_inx = int(above_th[-1])
    if fc_inx < 1:
        return 0.0

    f_sel = freqs_fc[: fc_inx + 1]
    mf_sel = mf_fc[: fc_inx + 1]
    wc = float(f_sel[-1])
    if wc <= 1e-12:
        return 0.0

    df = np.diff(f_sel) / wc
    dmf = np.diff(mf_sel)
    arc = -float(np.sum(np.sqrt(df * df + dmf * dmf)))
    return arc


def compute_ldlj(
    positions: np.ndarray,
    hz: float = 30.0,
) -> float:
    """Compute Log Dimensionless Jerk (LDLJ) over a multi-DOF trajectory (N, D).

    Values closer to 0 (less negative) indicate smoother motion.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1:
        pos = pos[:, None]
    if len(pos) < 4:
        return 0.0
    dt = 1.0 / float(hz)
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    speed_sq = np.sum(vel * vel, axis=1)
    v_peak_sq = float(np.max(speed_sq))
    if v_peak_sq <= 1e-12:
        return 0.0
    duration = (len(pos) - 1) * dt
    jerk_sq_integral = float(np.sum(np.sum(jerk * jerk, axis=1)) * dt)
    dlj = (duration**3 / v_peak_sq) * jerk_sq_integral
    if dlj <= 1e-15:
        return 0.0
    return -float(math.log(dlj))


def compute_spectral_features(
    signals: np.ndarray,
    hz: float = 30.0,
    cutoff_hz: float = 5.0,
) -> dict[str, Any]:
    """Compute Power Spectral Density (PSD), High-Frequency Energy Ratio (HFER), and Spectral Centroid."""
    arr = np.asarray(signals, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if len(arr) < 4:
        return {
            "high_freq_energy_ratio_hfer": 0.0,
            "spectral_centroid_hz": 0.0,
            "psd_freqs_hz": [],
            "psd_power": [],
        }

    dt = 1.0 / float(hz)
    # Compute spectrum on the velocity profile to remove static pose offset
    vel = np.gradient(arr, dt, axis=0)
    vel = vel - np.mean(vel, axis=0, keepdims=True)

    freqs = np.fft.rfftfreq(len(vel), d=dt)
    fft_vals = np.fft.rfft(vel, axis=0)
    power_per_bin = np.sum(np.abs(fft_vals) ** 2, axis=1) / float(len(vel))

    # Exclude DC (index 0) when computing dynamic frequency distribution
    ac_mask = freqs > 0.0
    if not np.any(ac_mask):
        return {
            "high_freq_energy_ratio_hfer": 0.0,
            "spectral_centroid_hz": 0.0,
            "psd_freqs_hz": freqs.tolist(),
            "psd_power": power_per_bin.tolist(),
        }

    ac_freqs = freqs[ac_mask]
    ac_power = power_per_bin[ac_mask]
    total_power = float(np.sum(ac_power))
    if total_power <= 1e-15:
        hfer = 0.0
        centroid = 0.0
    else:
        hf_mask = ac_freqs >= float(cutoff_hz)
        hfer = float(np.sum(ac_power[hf_mask]) / total_power)
        centroid = float(np.sum(ac_freqs * ac_power) / total_power)

    return {
        "high_freq_energy_ratio_hfer": hfer,
        "spectral_centroid_hz": centroid,
        "psd_freqs_hz": freqs.tolist(),
        "psd_power": power_per_bin.tolist(),
    }


def compute_chunk_boundary_jump(
    joint_trajectory: Sequence[Sequence[float]],
    chunk_size: int = 10,
    hz: float = 30.0,
) -> dict[str, float]:
    """Compute velocity and position discontinuities across action-chunk boundaries."""
    q = np.asarray(joint_trajectory, dtype=float)
    if q.ndim != 2 or len(q) <= chunk_size + 1:
        return {
            "chunk_boundary_jump_rad_s": 0.0,
            "max_chunk_boundary_jump_rad_s": 0.0,
        }
    dt = 1.0 / float(hz)
    vel = np.diff(q, axis=0) / dt
    boundary_jumps = []
    # Boundary occurs between step k*chunk_size - 1 and k*chunk_size
    for boundary_idx in range(chunk_size, len(q) - 1, chunk_size):
        v_before = vel[boundary_idx - 1]
        v_after = vel[boundary_idx]
        boundary_jumps.append(float(np.linalg.norm(v_after - v_before)))
    if not boundary_jumps:
        return {
            "chunk_boundary_jump_rad_s": 0.0,
            "max_chunk_boundary_jump_rad_s": 0.0,
        }
    return {
        "chunk_boundary_jump_rad_s": float(np.mean(boundary_jumps)),
        "max_chunk_boundary_jump_rad_s": float(np.max(boundary_jumps)),
    }


def compute_comprehensive_metrics(
    object_samples: Sequence[Sequence[float]],
    joint_samples: Sequence[Sequence[float]],
    ee_samples: Optional[Sequence[Sequence[float]]],
    destination_xy: Sequence[float],
    final_object_xyz: Sequence[float],
    collision_count: float,
    task_success: bool,
    hz: float = 30.0,
    chunk_size: int = 10,
    arm_joint_count: Optional[int] = None,
    model_latencies_s: Sequence[float] = (),
    spectrum_decode_latencies_s: Sequence[float] = (),
    compression_ratios: Sequence[float] = (),
    oracle_path_length_m: Optional[float] = None,
    grasp_slip_or_drop: bool = False,
    wall_time_s: float = 0.0,
) -> dict[str, Any]:
    """Compute the full Scope 2 evaluation matrix for a single rollout."""
    dt = 1.0 / float(hz)
    endpoint_error = float(
        np.linalg.norm(np.asarray(final_object_xyz[:2], dtype=float) - np.asarray(destination_xy[:2], dtype=float))
    )

    # 1. Object Cartesian Trajectory Metrics
    if len(object_samples) >= 4:
        obj_arr = np.asarray(object_samples, dtype=float)
        obj_xy = obj_arr[:, 1:3]
        obj_vel = np.gradient(obj_xy, dt, axis=0)
        obj_acc = np.gradient(obj_vel, dt, axis=0)
        obj_jerk = np.gradient(obj_acc, dt, axis=0)
        obj_speed = np.linalg.norm(obj_vel, axis=1)
        obj_jerk_norm = np.linalg.norm(obj_jerk, axis=1)
        path_length_m = float(np.linalg.norm(np.diff(obj_xy, axis=0), axis=1).sum())
        max_speed_m_s = float(np.max(obj_speed))
        max_acc_m_s2 = float(np.max(np.linalg.norm(obj_acc, axis=1)))
        object_max_jerk = float(np.max(obj_jerk_norm))
        object_rms_jerk = float(np.sqrt(np.mean(obj_jerk_norm**2)))
        height_range_m = float(np.ptp(obj_arr[:, 3]))
    elif len(object_samples) >= 2:
        obj_arr = np.asarray(object_samples, dtype=float)
        obj_xy = obj_arr[:, 1:3]
        path_length_m = float(np.linalg.norm(np.diff(obj_xy, axis=0), axis=1).sum())
        max_speed_m_s = 0.0
        max_acc_m_s2 = 0.0
        object_max_jerk = 0.0
        object_rms_jerk = 0.0
        height_range_m = float(np.ptp(obj_arr[:, 3]))
    else:
        path_length_m = 0.0
        max_speed_m_s = 0.0
        max_acc_m_s2 = 0.0
        object_max_jerk = 0.0
        object_rms_jerk = 0.0
        height_range_m = 0.0

    # 2. Joint-Space Smoothness & Spectral Metrics
    if len(joint_samples) >= 4:
        q_arr = np.asarray(joint_samples, dtype=float)
        if arm_joint_count is not None and q_arr.shape[1] >= arm_joint_count:
            arm_q = q_arr[:, :arm_joint_count]
        else:
            arm_q = q_arr[:, :-1] if q_arr.shape[1] > 1 else q_arr
        dq = np.gradient(arm_q, dt, axis=0)
        ddq = np.gradient(dq, dt, axis=0)
        dddq = np.gradient(ddq, dt, axis=0)
        q_jerk_norm = np.linalg.norm(dddq, axis=1)
        joint_rms_jerk = float(np.sqrt(np.mean(q_jerk_norm**2)))
        joint_max_jerk = float(np.max(q_jerk_norm))
        joint_speed = np.linalg.norm(dq, axis=1)
        sparc_val = compute_sparc(joint_speed, hz=hz)
        ldlj_val = compute_ldlj(arm_q, hz=hz)
        spectral_info = compute_spectral_features(arm_q, hz=hz, cutoff_hz=min(5.0, hz * 0.25))
        boundary_info = compute_chunk_boundary_jump(arm_q, chunk_size=chunk_size, hz=hz)
    else:
        joint_rms_jerk = 0.0
        joint_max_jerk = 0.0
        sparc_val = 0.0
        ldlj_val = 0.0
        spectral_info = {
            "high_freq_energy_ratio_hfer": 0.0,
            "spectral_centroid_hz": 0.0,
            "psd_freqs_hz": [],
            "psd_power": [],
        }
        boundary_info = {
            "chunk_boundary_jump_rad_s": 0.0,
            "max_chunk_boundary_jump_rad_s": 0.0,
        }

    # 3. End-Effector 3D Smoothness Metrics
    if ee_samples is not None and len(ee_samples) >= 4:
        ee_arr = np.asarray(ee_samples, dtype=float)
        ee_xyz = ee_arr[:, 1:4] if ee_arr.shape[1] >= 4 else ee_arr[:, :3]
        ee_vel = np.gradient(ee_xyz, dt, axis=0)
        ee_acc = np.gradient(ee_vel, dt, axis=0)
        ee_jerk = np.gradient(ee_acc, dt, axis=0)
        ee_jerk_norm = np.linalg.norm(ee_jerk, axis=1)
        ee_rms_jerk = float(np.sqrt(np.mean(ee_jerk_norm**2)))
        ee_max_jerk = float(np.max(ee_jerk_norm))
    else:
        ee_rms_jerk = object_rms_jerk
        ee_max_jerk = object_max_jerk

    # 4. Path Efficiency & Latency
    if oracle_path_length_m and oracle_path_length_m > 1e-6 and path_length_m > 1e-6:
        path_efficiency = float(min(1.0, oracle_path_length_m / path_length_m))
    else:
        path_efficiency = 1.0 if task_success and path_length_m > 1e-6 else 0.0

    mean_inf_ms = float(np.mean(model_latencies_s) * 1000.0) if model_latencies_s else 0.0
    total_inf_s = float(np.sum(model_latencies_s)) if model_latencies_s else 0.0
    mean_dec_ms = float(np.mean(spectrum_decode_latencies_s) * 1000.0) if spectrum_decode_latencies_s else 0.0
    mean_comp_ratio = float(np.mean(compression_ratios)) if compression_ratios else 1.0

    return {
        # Group A: Task Accuracy & Safety
        "task_success": bool(task_success),
        "endpoint_error_m": endpoint_error,
        "collision_count": float(collision_count),
        "grasp_slip_or_drop": bool(grasp_slip_or_drop),
        "path_length_m": path_length_m,
        "path_efficiency_ratio": path_efficiency,
        "height_range_m": height_range_m,
        # Group B: Time-Domain Smoothness
        "max_speed_m_s": max_speed_m_s,
        "max_acceleration_m_s2": max_acc_m_s2,
        "max_jerk_m_s3": object_max_jerk,
        "object_max_jerk_m_s3": object_max_jerk,
        "object_rms_jerk_m_s3": object_rms_jerk,
        "ee_rms_jerk_m_s3": ee_rms_jerk,
        "ee_max_jerk_m_s3": ee_max_jerk,
        "joint_rms_jerk_rad_s3": joint_rms_jerk,
        "joint_max_jerk_rad_s3": joint_max_jerk,
        "chunk_boundary_jump_rad_s": boundary_info["chunk_boundary_jump_rad_s"],
        "max_chunk_boundary_jump_rad_s": boundary_info["max_chunk_boundary_jump_rad_s"],
        "dimensionless_jerk_ldlj": ldlj_val,
        # Group C: Frequency-Domain Smoothness
        "sparc_smoothness": sparc_val,
        "high_freq_energy_ratio_hfer": spectral_info["high_freq_energy_ratio_hfer"],
        "spectral_centroid_hz": spectral_info["spectral_centroid_hz"],
        "token_compression_ratio": mean_comp_ratio,
        # Group D: Latency & Timing
        "mean_model_inference_ms": mean_inf_ms,
        "inference_latency_s": total_inf_s,
        "mean_chunk_latency_s": mean_inf_ms / 1000.0,
        "mean_spectrum_decode_ms": mean_dec_ms,
        "total_control_latency_ms": mean_inf_ms + mean_dec_ms,
        "episode_wall_time_s": float(wall_time_s),
        # PSD curve for diagnostic plotting
        "psd_freqs_hz": spectral_info["psd_freqs_hz"],
        "psd_power": spectral_info["psd_power"],
    }


def save_spectral_diagnostic_chart(
    path: Path,
    object_samples: Sequence[Sequence[float]],
    joint_samples: Sequence[Sequence[float]],
    obstacles: Sequence[Sequence[float]],
    metrics: dict[str, Any],
    title: str,
    workspace: Sequence[float] = (-0.24, -0.20, 0.24, 0.20),
    oracle_xy: Optional[Sequence[Sequence[float]]] = None,
    chunk_size: int = 10,
) -> None:
    """Save a 4-panel diagnostic chart: Kinematics/Jerk + Joint Angles + PSD Spectrum + 2D Workspace Path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1200, 920), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 14), title, fill="black")

    summary_line = (
        f"Success={metrics.get('task_success')} | Err={metrics.get('endpoint_error_m', 0):.3f}m | "
        f"JointRMSJerk={metrics.get('joint_rms_jerk_rad_s3', 0):.1f} | "
        f"HFER={metrics.get('high_freq_energy_ratio_hfer', 0):.3f} | "
        f"SPARC={metrics.get('sparc_smoothness', 0):.2f} | "
        f"ChunkJump={metrics.get('chunk_boundary_jump_rad_s', 0):.2f} rad/s"
    )
    draw.text((35, 34), summary_line, fill=(55, 55, 55))

    if not object_samples or len(object_samples) < 2:
        draw.text((40, 75), "No executed trajectory samples", fill=(180, 30, 30))
        canvas.save(path)
        return

    values = np.asarray(object_samples, dtype=float)
    times, xy = values[:, 0], values[:, 1:3]
    dt = float(times[1] - times[0]) if len(times) > 1 and times[1] > times[0] else 1.0 / 30.0
    speed = np.linalg.norm(np.gradient(xy, dt, axis=0), axis=1)
    jerk = np.linalg.norm(
        np.gradient(np.gradient(np.gradient(xy, dt, axis=0), dt, axis=0), dt, axis=0),
        axis=1,
    )

    # Panel 1 & 2: Speed & Jerk with chunk-boundary vertical lines
    for panel, (series, label, color) in enumerate(
        ((speed, "Planar Speed (m/s)", (30, 95, 190)), (jerk, "Cartesian Jerk (m/s^3)", (205, 50, 45)))
    ):
        left, right = 65, 1150
        top, bottom = 65 + panel * 165, 195 + panel * 165
        draw.rectangle((left, top, right, bottom), outline="black")
        draw.text((left + 6, top + 4), label, fill=color)
        maximum = max(float(np.max(series)), 1e-9)
        duration = max(float(times[-1]), dt)
        # Draw chunk boundaries
        for step_idx in range(chunk_size, len(times), chunk_size):
            bx = left + (right - left) * float(times[step_idx]) / duration
            draw.line((bx, top + 1, bx, bottom - 1), fill=(220, 220, 220), width=1)
        pts = [
            (
                left + (right - left) * float(t) / duration,
                bottom - (bottom - top) * float(v) / maximum,
            )
            for t, v in zip(times, series)
        ]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)

    # Panel 3 (Middle Left): Joint Trajectories
    j_left, j_right, j_top, j_bottom = 65, 585, 405, 625
    draw.rectangle((j_left, j_top, j_right, j_bottom), outline="black")
    draw.text((j_left + 6, j_top + 4), "Joint Trajectories q(t) [rad]", fill="black")
    if joint_samples:
        j_arr = np.asarray(joint_samples, dtype=float)
        palette = ("red", "blue", "green", "purple", "orange", "brown")
        j_min = float(j_arr.min())
        j_span = max(float(j_arr.max()) - j_min, 1e-9)
        for j_idx in range(j_arr.shape[1]):
            pts = [
                (
                    j_left + (j_right - j_left) * s / max(len(j_arr) - 1, 1),
                    j_bottom - (j_bottom - j_top - 22) * (float(val) - j_min) / j_span,
                )
                for s, val in enumerate(j_arr[:, j_idx])
            ]
            if len(pts) > 1:
                draw.line(pts, fill=palette[j_idx % len(palette)], width=2)
            draw.text(
                (j_left + 8 + j_idx * 75, j_bottom - 16),
                f"q{j_idx + 1}",
                fill=palette[j_idx % len(palette)],
            )

    # Panel 4 (Middle Right): Power Spectral Density (PSD)
    p_left, p_right, p_top, p_bottom = 630, 1150, 405, 625
    draw.rectangle((p_left, p_top, p_right, p_bottom), outline="black")
    draw.text(
        (p_left + 6, p_top + 4),
        f"Joint Velocity Power Spectrum (HFER={metrics.get('high_freq_energy_ratio_hfer', 0):.2%})",
        fill=(20, 120, 90),
    )
    psd_freqs = metrics.get("psd_freqs_hz") or []
    psd_power = metrics.get("psd_power") or []
    if len(psd_freqs) > 1 and len(psd_power) == len(psd_freqs):
        f_arr = np.asarray(psd_freqs, dtype=float)
        p_arr = np.asarray(psd_power, dtype=float)
        max_f = max(float(f_arr[-1]), 1.0)
        max_p = max(float(np.max(p_arr[1:])) if len(p_arr) > 1 else float(np.max(p_arr)), 1e-9)
        cutoff_x = p_left + (p_right - p_left) * min(5.0, max_f) / max_f
        draw.line((cutoff_x, p_top + 20, cutoff_x, p_bottom - 1), fill=(220, 100, 100), width=2)
        draw.text((cutoff_x + 4, p_top + 22), "5 Hz cutoff", fill=(200, 60, 60))
        pts = [
            (
                p_left + (p_right - p_left) * float(f) / max_f,
                p_bottom - (p_bottom - p_top - 24) * min(1.0, float(pw) / max_p),
            )
            for f, pw in zip(f_arr[1:], p_arr[1:])
        ]
        if len(pts) > 1:
            draw.line(pts, fill=(20, 120, 90), width=3)

    # Panel 5 (Bottom): 2D Workspace XY Path + Obstacles + Oracle Reference
    w_left, w_right, w_top, w_bottom = 65, 1150, 655, 895
    draw.rectangle((w_left, w_top, w_right, w_bottom), outline="black")
    draw.text((w_left + 6, w_top + 4), "2D Workspace XY Trajectory & Obstacles", fill="black")
    xmin, ymin, xmax, ymax = map(float, workspace)
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)

    def project(pt: Sequence[float]) -> tuple[float, float]:
        return (
            w_left + (float(pt[0]) - xmin) * (w_right - w_left) / xspan,
            w_bottom - (float(pt[1]) - ymin) * (w_bottom - w_top) / yspan,
        )

    for bounds in obstacles:
        draw.rectangle(
            (*project((bounds[0], bounds[3])), *project((bounds[2], bounds[1]))),
            fill=(145, 145, 145),
        )
    if oracle_xy is not None and len(oracle_xy) > 1:
        oracle_pts = [project(pt) for pt in oracle_xy]
        draw.line(oracle_pts, fill=(130, 180, 130), width=2)
    exec_pts = [project(pt) for pt in xy]
    if len(exec_pts) > 1:
        draw.line(exec_pts, fill=(115, 40, 165), width=3)
    for pt, col in ((exec_pts[0], (30, 145, 85)), (exec_pts[-1], (215, 80, 45))):
        draw.ellipse((pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5), fill=col)

    canvas.save(path)

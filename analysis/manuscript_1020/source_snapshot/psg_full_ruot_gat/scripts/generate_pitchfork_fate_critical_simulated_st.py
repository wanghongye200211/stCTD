#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


CELL_TYPES = ["Prog", "FateA", "FateB"]
COLORS = {"Prog": "#4C78A8", "FateA": "#59A14F", "FateB": "#E15759"}
DEFAULT_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]


def sigmoid(x: np.ndarray | float, tau: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float) / float(tau)))


def gaussian(x: np.ndarray | float, center: float, sigma: float) -> np.ndarray:
    return np.exp(-((np.asarray(x, dtype=float) - float(center)) ** 2) / (2.0 * float(sigma) ** 2))


def smooth_progress(time: np.ndarray | float, critical_time: float, onset_time: float, tau: float) -> np.ndarray:
    raw = sigmoid(np.asarray(time, dtype=float) - critical_time, tau)
    start = float(sigmoid(onset_time - critical_time, tau))
    progress = (raw - start) / max(1.0 - start, 1e-6)
    return np.clip(progress, 0.0, 1.0)


def smooth_rate(time: np.ndarray | float, critical_time: float, sigma: float) -> np.ndarray:
    return gaussian(time, critical_time, sigma)


def pitchfork_r(time: np.ndarray | float, critical_time: float, width: float) -> np.ndarray:
    # r crosses zero at the ground-truth critical time.
    return 1.75 * np.tanh((np.asarray(time, dtype=float) - critical_time) / float(width))


def sample_rotated_ellipse(
    rng: np.random.Generator,
    n: int,
    center: tuple[float, float],
    axes: tuple[float, float],
    angle: float = 0.0,
) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, int(n))
    radius = np.sqrt(rng.uniform(0.0, 1.0, int(n)))
    local = np.column_stack(
        [
            float(axes[0]) * radius * np.cos(theta),
            float(axes[1]) * radius * np.sin(theta),
        ]
    )
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.asarray([[c, -s], [s, c]], dtype=float)
    return (local @ rot.T + np.asarray(center, dtype=float)).astype(np.float32)


def sample_regular_spatial(
    rng: np.random.Generator,
    cell_type: str,
    n: int,
    hidden_fate: np.ndarray,
    progress: float,
    time: float,
    critical_time: float,
) -> np.ndarray:
    """Sample from a smooth, compact tissue template instead of diffuse Gaussian blobs."""
    p = float(np.clip(progress, 0.0, 1.0))
    t_shift = 0.015 * (float(time) - float(critical_time))
    if cell_type == "Prog":
        xy = sample_rotated_ellipse(
            rng,
            n,
            center=(0.0, -0.12 + t_shift + 0.02 * p),
            axes=(0.46 - 0.05 * p, 0.58 - 0.08 * p),
            angle=0.03,
        )
        # Early progenitors carry hidden fate, but the tissue remains a single smooth domain.
        xy[:, 0] += (0.030 + 0.030 * p) * hidden_fate.astype(np.float32)
        xy[:, 1] += 0.035 * (1.0 - (xy[:, 0] / max(0.46 - 0.05 * p, 1e-6)) ** 2)
    elif cell_type == "FateA":
        branch = 0.22 + 0.36 * p
        xy = sample_rotated_ellipse(
            rng,
            n,
            center=(-branch, 0.05 + t_shift + 0.20 * p),
            axes=(0.22 + 0.02 * p, 0.34 - 0.02 * p),
            angle=-0.42,
        )
    elif cell_type == "FateB":
        branch = 0.24 + 0.40 * p
        xy = sample_rotated_ellipse(
            rng,
            n,
            center=(branch, 0.06 + t_shift + 0.19 * p),
            axes=(0.25 + 0.02 * p, 0.36 - 0.02 * p),
            angle=0.38,
        )
    else:
        raise ValueError(cell_type)
    # A tiny jitter avoids perfectly synthetic lattice-like boundaries without creating outliers.
    xy += rng.normal(0.0, 0.010, xy.shape).astype(np.float32)
    return xy.astype(np.float32)


def parse_float_sequence(text: str, expected: int, name: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(values) != int(expected):
        raise ValueError(f"{name} must contain {expected} comma-separated values; got {len(values)}.")
    return values


def rotate_translate_slice(xy: np.ndarray, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    center = np.asarray(xy, dtype=np.float32).mean(axis=0, keepdims=True)
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    shifted = (np.asarray(xy, dtype=np.float32) - center) @ rot.T
    return (shifted + center + np.asarray([[float(tx), float(ty)]], dtype=np.float32)).astype(np.float32)


def add_slice_misalignment(
    adata: ad.AnnData,
    rotation_deg: list[float],
    translation_x: list[float],
    translation_y: list[float],
) -> None:
    aligned = np.asarray(adata.obsm["spatial"], dtype=np.float32).copy()
    raw = aligned.copy()
    times = sorted(float(t) for t in adata.obs["time"].unique())
    records = []
    for i, time in enumerate(times):
        mask = np.isclose(adata.obs["time"].astype(float), time)
        raw[mask] = rotate_translate_slice(aligned[mask], rotation_deg[i], translation_x[i], translation_y[i])
        records.append(
            {
                "time": float(time),
                "rotation_deg": float(rotation_deg[i]),
                "translation_x": float(translation_x[i]),
                "translation_y": float(translation_y[i]),
            }
        )
    adata.obsm["spatial_aligned"] = aligned
    adata.obsm["spatial_misaligned"] = raw
    adata.obsm["spatial_raw"] = raw
    adata.uns["slice_misalignment"] = {
        "enabled": True,
        "kind": "synthetic per-slice rigid rotation plus translation; no scaling is applied",
        "records": records,
        "model_uses": "obsm['X_spatial_input'] is computed from aligned coordinates; obsm['spatial_raw'] stores misaligned raw coordinates for alignment demonstration.",
    }


def total_cells(
    base_cells: int,
    time: float,
    critical_time: float,
    mass_linear_slope: float,
    mass_transition_boost: float,
    mass_wave_amp: float = 0.0,
    mass_wave_period: float = 2.8,
    mass_wave_phase: float = 0.0,
    mass_dip_amp: float = 0.0,
    mass_dip_time: float | None = None,
    mass_dip_sigma: float = 0.5,
    mass_bump_amp: float = 0.0,
    mass_bump_time: float | None = None,
    mass_bump_sigma: float = 0.5,
) -> int:
    growth_phase = float(sigmoid(time - critical_time, 0.9))
    factor = 1.00 + float(mass_linear_slope) * time + float(mass_transition_boost) * growth_phase
    if abs(float(mass_wave_amp)) > 0:
        factor += float(mass_wave_amp) * np.sin(2.0 * np.pi * (float(time) - float(mass_wave_phase)) / float(mass_wave_period))
    if abs(float(mass_dip_amp)) > 0:
        center = critical_time if mass_dip_time is None else float(mass_dip_time)
        factor -= float(mass_dip_amp) * float(gaussian(time, center, mass_dip_sigma))
    if abs(float(mass_bump_amp)) > 0:
        center = critical_time if mass_bump_time is None else float(mass_bump_time)
        factor += float(mass_bump_amp) * float(gaussian(time, center, mass_bump_sigma))
    return int(round(base_cells * max(factor, 0.25)))


def type_counts(
    n_total: int,
    time: float,
    critical_time: float,
    onset_time: float,
    tau: float,
    min_terminal_fraction: float,
    max_terminal_fraction: float,
    terminal_fraction_power: float,
    terminal_fatea_fraction: float,
) -> dict[str, int]:
    progress = float(smooth_progress(time, critical_time, onset_time, tau))
    min_terminal = float(np.clip(min_terminal_fraction, 0.0, max_terminal_fraction))
    max_terminal = float(max_terminal_fraction)
    terminal_fraction = min_terminal + (max_terminal - min_terminal) * progress ** float(terminal_fraction_power)
    prog_fraction = max(1.0 - terminal_fraction, 0.08)
    terminal_fraction = 1.0 - prog_fraction
    fatea_fraction = float(np.clip(terminal_fatea_fraction, 0.05, 0.95)) * terminal_fraction
    fateb_fraction = terminal_fraction - fatea_fraction
    frac = np.asarray([prog_fraction, fatea_fraction, fateb_fraction], dtype=float)
    counts = np.floor(frac / frac.sum() * int(n_total)).astype(int)
    counts[-1] += int(n_total) - int(counts.sum())
    return dict(zip(CELL_TYPES, counts.tolist()))


def sample_stage(
    rng: np.random.Generator,
    time: float,
    n_total: int,
    critical_time: float,
    onset_time: float,
    tau: float,
    min_terminal_fraction: float,
    max_terminal_fraction: float,
    terminal_fraction_power: float,
    terminal_fatea_fraction: float,
    growth_scale: float,
    global_y_slope: float,
    spatial_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = type_counts(
        n_total,
        time,
        critical_time,
        onset_time,
        tau,
        min_terminal_fraction,
        max_terminal_fraction,
        terminal_fraction_power,
        terminal_fatea_fraction,
    )
    progress = float(smooth_progress(time, critical_time, onset_time, tau))
    rate = float(smooth_rate(time, critical_time, 0.55))
    r_value = float(pitchfork_r(time, critical_time, 0.62))
    branch_len = 0.62 * progress ** 1.35
    gene_branch_strength = progress ** 1.70
    trunk_y = -0.55 + float(global_y_slope) * (float(time) - float(critical_time))

    obs_rows: list[dict[str, object]] = []
    spatial_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    growth_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    hidden_parts: list[np.ndarray] = []

    for cell_type in CELL_TYPES:
        n = int(counts[cell_type])
        if n <= 0:
            continue
        if cell_type == "Prog":
            hidden_fate = rng.choice([-1.0, 1.0], size=n)
            branch_gene = rng.normal(0.10 * hidden_fate * gene_branch_strength, 0.13 + 0.04 * progress, n)
            if spatial_mode == "regular":
                xy_regular = sample_regular_spatial(rng, cell_type, n, hidden_fate, progress, time, critical_time)
                x = xy_regular[:, 0]
                y = xy_regular[:, 1]
            else:
                x = rng.normal(0.09 * hidden_fate * progress, 0.18, n)
                y = rng.normal(trunk_y + 0.03 * progress, 0.18, n)
            v_x = hidden_fate * (0.10 * rate + 0.03 * progress)
            v_y = np.full(n, (0.02 if spatial_mode == "regular" else 0.11) + 0.10 * rate, dtype=float)
            growth = np.full(n, -0.015 - 0.040 * rate - 0.012 * progress, dtype=float) * float(growth_scale)
        elif cell_type == "FateA":
            hidden_fate = np.full(n, -1.0)
            branch_gene = rng.normal(-(1.65 * gene_branch_strength), 0.16 + 0.04 * (1.0 - progress), n)
            if spatial_mode == "regular":
                xy_regular = sample_regular_spatial(rng, cell_type, n, hidden_fate, progress, time, critical_time)
                x = xy_regular[:, 0]
                y = xy_regular[:, 1]
            else:
                x = rng.normal(-branch_len, 0.22, n)
                y = rng.normal(trunk_y + 0.14 * progress + 0.04 * branch_len, 0.20, n)
            v_x = np.full(n, -(0.28 * rate + 0.20 * progress + 0.04), dtype=float)
            v_y = np.full(n, 0.06 + 0.28 * rate + 0.07 * progress, dtype=float)
            growth = np.full(n, 0.035 + 0.115 * rate + 0.018 * progress, dtype=float) * float(growth_scale)
        elif cell_type == "FateB":
            hidden_fate = np.full(n, 1.0)
            branch_gene = rng.normal(1.65 * gene_branch_strength, 0.16 + 0.04 * (1.0 - progress), n)
            if spatial_mode == "regular":
                xy_regular = sample_regular_spatial(rng, cell_type, n, hidden_fate, progress, time, critical_time)
                x = xy_regular[:, 0]
                y = xy_regular[:, 1]
            else:
                x = rng.normal(branch_len, 0.22, n)
                y = rng.normal(trunk_y + 0.14 * progress + 0.04 * branch_len, 0.20, n)
            v_x = np.full(n, 0.28 * rate + 0.20 * progress + 0.04, dtype=float)
            v_y = np.full(n, 0.06 + 0.28 * rate + 0.07 * progress, dtype=float)
            growth = np.full(n, 0.035 + 0.115 * rate + 0.018 * progress, dtype=float) * float(growth_scale)
        else:
            raise ValueError(cell_type)

        xy = np.column_stack([x, y]).astype(np.float32)
        if spatial_mode != "regular":
            xy[:, 1] += 0.10 * np.abs(xy[:, 0])
            xy += rng.normal(0.0, 0.040, xy.shape).astype(np.float32)
        velocity = np.column_stack([v_x, v_y]).astype(np.float32)
        branch_gene = branch_gene.astype(np.float32)
        hidden_fate = hidden_fate.astype(np.float32)

        start = len(obs_rows)
        for i in range(n):
            terminal = "FateA" if hidden_fate[i] < 0 else "FateB"
            obs_rows.append(
                {
                    "cell_id": f"t{time:g}_{cell_type}_{start + i:05d}",
                    "time": float(time),
                    "time_input": float(time),
                    "Batch": f"t{time:g}",
                    "sample_id": f"PITCHFORK_t{time:g}",
                    "section_id": f"PITCHFORK_t{time:g}",
                    "Annotation": cell_type,
                    "cell_type": cell_type,
                    "lineage_prior": f"lineage_{terminal}",
                    "terminal_fate": terminal,
                    "critical_progress": progress,
                    "critical_rate": rate,
                    "pitchfork_r": r_value,
                    "is_ground_truth_critical_time": bool(abs(float(time) - critical_time) < 1e-8),
                }
            )
        spatial_parts.append(xy)
        velocity_parts.append(velocity)
        growth_parts.append(growth.astype(np.float32))
        branch_parts.append(branch_gene)
        hidden_parts.append(hidden_fate)

    obs = pd.DataFrame(obs_rows)
    spatial = np.vstack(spatial_parts).astype(np.float32)
    velocity = np.vstack(velocity_parts).astype(np.float32)
    growth = np.concatenate(growth_parts).astype(np.float32)
    branch = np.concatenate(branch_parts).astype(np.float32)
    hidden = np.concatenate(hidden_parts).astype(np.float32)
    return obs, spatial, velocity, growth, branch, hidden


def build_latent(
    rng: np.random.Generator,
    obs: pd.DataFrame,
    spatial: np.ndarray,
    branch_gene: np.ndarray,
    hidden_fate: np.ndarray,
    critical_time: float,
) -> np.ndarray:
    time = obs["time"].to_numpy(float)
    labels = obs["Annotation"].to_numpy(str)
    progress = smooth_progress(time, critical_time, 0.75, 0.62).astype(np.float32)
    rate = smooth_rate(time, critical_time, 0.55).astype(np.float32)
    xy_norm = (spatial - spatial.mean(axis=0, keepdims=True)) / spatial.std(axis=0, keepdims=True).clip(1e-6)
    z = np.zeros((obs.shape[0], 10), dtype=np.float32)
    z[:, 0] = branch_gene
    z[:, 1] = progress
    z[:, 2] = rate
    z[:, 3] = xy_norm[:, 0]
    z[:, 4] = xy_norm[:, 1]
    z[:, 5] = hidden_fate * progress ** 1.40
    z[:, 6] = (labels != "Prog").astype(np.float32) * progress
    z[:, 7] = (time / max(float(time.max()), 1.0)).astype(np.float32)
    z[:, 8] = np.abs(branch_gene)
    z[:, 9] = np.asarray(obs["pitchfork_r"], dtype=np.float32)
    z += rng.normal(0.0, 0.045, z.shape).astype(np.float32)
    z = (z - z.mean(axis=0, keepdims=True)) / z.std(axis=0, keepdims=True).clip(1e-6)
    return z.astype(np.float32)


def make_counts_from_latent(
    rng: np.random.Generator,
    obs: pd.DataFrame,
    latent: np.ndarray,
    n_genes: int,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    labels = obs["Annotation"].to_numpy(str)
    terminal = obs["terminal_fate"].to_numpy(str)
    progress = np.asarray(obs["critical_progress"], dtype=float)
    terminal_strength = np.clip(progress, 0.0, 1.0) ** 1.45
    decoder = rng.normal(0.0, 0.22, size=(latent.shape[1], n_genes)).astype(np.float64)
    log_mu = -2.10 + np.einsum("nd,dg->ng", latent.astype(np.float64), decoder, optimize=True)

    block = max(60, min(120, n_genes // 8))
    for i, cell_type in enumerate(CELL_TYPES):
        start = i * block
        stop = min(start + block, n_genes)
        mask = labels == cell_type
        if np.any(mask):
            if cell_type == "Prog":
                strength = (2.05 - 0.35 * progress[mask])[:, None]
            else:
                strength = (2.15 * terminal_strength[mask])[:, None]
            log_mu[np.ix_(mask, np.arange(start, stop))] += strength

    if n_genes >= 620:
        rate = np.asarray(obs["critical_rate"], dtype=float)
        log_mu[:, 360:430] += (1.05 * progress)[:, None]
        log_mu[:, 430:500] += (1.40 * (labels == "FateA") * terminal_strength)[:, None]
        log_mu[:, 500:570] += (1.40 * (labels == "FateB") * terminal_strength)[:, None]
        log_mu[:, 570:620] += (1.10 * rate)[:, None]

    log_mu = np.clip(np.nan_to_num(log_mu, nan=-6.0, neginf=-6.0, posinf=3.2), -6.0, 3.2)
    lib = rng.lognormal(mean=np.log(2200), sigma=0.22, size=obs.shape[0]).astype(np.float32)
    mu = np.exp(log_mu)
    mu = mu / mu.sum(axis=1, keepdims=True).clip(1e-8) * lib[:, None]
    counts = rng.poisson(mu).astype(np.int32)
    denom = counts.sum(axis=1, keepdims=True).clip(1)
    lognorm = np.log1p(counts / denom * 1e4).astype(np.float32)
    centered = lognorm - lognorm.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    x_pca = (u[:, :10] * s[:10]).astype(np.float32)
    x_pca = (x_pca - x_pca.mean(axis=0, keepdims=True)) / x_pca.std(axis=0, keepdims=True).clip(1e-6)
    return sp.csr_matrix(counts), lognorm, x_pca.astype(np.float32)


def true_potential(obs: pd.DataFrame, branch_gene: np.ndarray, spatial: np.ndarray, critical_time: float) -> np.ndarray:
    time = obs["time"].to_numpy(float)
    t_min = float(np.min(time))
    t_max = float(np.max(time))
    t_scaled = (time - t_min) / max(t_max - t_min, 1e-6)
    progress = np.asarray(obs["critical_progress"], dtype=float)
    rate = np.asarray(obs["critical_rate"], dtype=float)
    r = pitchfork_r(obs["time"].to_numpy(float), critical_time, 0.62)
    b = branch_gene.astype(float)
    # Developmental potential should mostly decrease with time/commitment. The
    # pitchfork term only shapes local fate basins; it should not create a
    # global potential peak at the critical time.
    developmental_baseline = 1.15 - 0.78 * t_scaled
    commitment_drop = -0.20 * np.abs(b) * progress
    pitch = 0.035 * (0.25 * b**4 - 0.50 * r * b**2)
    xy = (spatial - spatial.mean(axis=0, keepdims=True)) / spatial.std(axis=0, keepdims=True).clip(1e-6)
    spatial_term = 0.012 * (xy[:, 0] ** 2 + 0.7 * xy[:, 1] ** 2)
    transition_activity = 0.025 * rate
    return (developmental_baseline + commitment_drop + pitch + spatial_term + transition_activity).astype(np.float32)


def build_adata(
    rng: np.random.Generator,
    times: list[float],
    base_cells: int,
    n_genes: int,
    critical_time: float,
    onset_time: float,
    tau: float,
    max_terminal_fraction: float,
    min_terminal_fraction: float,
    terminal_fraction_power: float,
    terminal_fatea_fraction: float,
    gene_input: str,
    mass_linear_slope: float,
    mass_transition_boost: float,
    mass_wave_amp: float,
    mass_wave_period: float,
    mass_wave_phase: float,
    mass_dip_amp: float,
    mass_dip_time: float | None,
    mass_dip_sigma: float,
    mass_bump_amp: float,
    mass_bump_time: float | None,
    mass_bump_sigma: float,
    growth_scale: float,
    global_y_slope: float,
    spatial_mode: str,
) -> ad.AnnData:
    obs_parts: list[pd.DataFrame] = []
    spatial_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    growth_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    hidden_parts: list[np.ndarray] = []
    for time in times:
        n_total = total_cells(
            base_cells,
            time,
            critical_time,
            mass_linear_slope,
            mass_transition_boost,
            mass_wave_amp,
            mass_wave_period,
            mass_wave_phase,
            mass_dip_amp,
            mass_dip_time,
            mass_dip_sigma,
            mass_bump_amp,
            mass_bump_time,
            mass_bump_sigma,
        )
        obs, spatial, velocity, growth, branch, hidden = sample_stage(
            rng,
            time,
            n_total,
            critical_time,
            onset_time,
            tau,
            min_terminal_fraction,
            max_terminal_fraction,
            terminal_fraction_power,
            terminal_fatea_fraction,
            growth_scale,
            global_y_slope,
            spatial_mode,
        )
        obs_parts.append(obs)
        spatial_parts.append(spatial)
        velocity_parts.append(velocity)
        growth_parts.append(growth)
        branch_parts.append(branch)
        hidden_parts.append(hidden)

    obs = pd.concat(obs_parts, axis=0, ignore_index=True)
    obs.index = obs["cell_id"].astype(str)
    spatial = np.vstack(spatial_parts).astype(np.float32)
    velocity = np.vstack(velocity_parts).astype(np.float32)
    growth = np.concatenate(growth_parts).astype(np.float32)
    branch = np.concatenate(branch_parts).astype(np.float32)
    hidden = np.concatenate(hidden_parts).astype(np.float32)
    latent = build_latent(rng, obs, spatial, branch, hidden, critical_time)
    counts, lognorm, x_pca = make_counts_from_latent(rng, obs, latent, n_genes)

    obs["Annotation"] = pd.Categorical(obs["Annotation"], categories=CELL_TYPES)
    obs["cell_type"] = pd.Categorical(obs["cell_type"], categories=CELL_TYPES)
    obs["lineage_prior"] = pd.Categorical(obs["lineage_prior"], categories=["lineage_FateA", "lineage_FateB"])
    obs["terminal_fate"] = pd.Categorical(obs["terminal_fate"], categories=["FateA", "FateB"])
    obs["true_growth_rate"] = growth
    obs["true_branch_gene"] = branch
    obs["hidden_fate_direction"] = hidden
    obs["true_potential"] = true_potential(obs, branch, spatial, critical_time)
    grad_b = branch.astype(float) ** 3 - np.asarray(obs["pitchfork_r"], dtype=float) * branch.astype(float)
    obs["true_potential_grad_b_abs"] = np.abs(grad_b).astype(np.float32)
    obs["critical_score"] = (np.asarray(obs["critical_rate"], dtype=float) / (1.0 + np.abs(grad_b))).astype(np.float32)

    var = pd.DataFrame(index=[f"Gene{i:04d}" for i in range(n_genes)])
    var["gene_ids"] = var.index
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    adata.layers["lognorm"] = sp.csr_matrix(lognorm)
    adata.obsm["spatial"] = spatial
    adata.obsm["spatial_raw"] = spatial.copy()
    spatial_input = (spatial - spatial.mean(axis=0, keepdims=True)) / spatial.std(axis=0, keepdims=True).clip(1e-6)
    adata.obsm["X_spatial_input"] = spatial_input.astype(np.float32)
    adata.obsm["spatial_globalnorm"] = spatial_input.astype(np.float32)
    adata.obsm["X_true_latent"] = latent.astype(np.float32)
    adata.obsm["X_pca"] = x_pca.astype(np.float32)
    adata.obsm["X_pca_from_counts"] = x_pca.astype(np.float32)
    adata.obsm["X_gene_input"] = (latent if gene_input == "true_latent" else x_pca).astype(np.float32)
    adata.obsm["true_spatial_velocity"] = velocity
    adata.uns["Annotation_colors"] = [COLORS[c] for c in CELL_TYPES]
    adata.uns["cell_type_colors"] = [COLORS[c] for c in CELL_TYPES]
    adata.uns["simulation"] = {
        "name": "Pitchfork fate critical point simulation",
        "cell_types": CELL_TYPES,
        "critical_time": float(critical_time),
        "onset_time": float(onset_time),
        "transition_tau": float(tau),
        "min_terminal_fraction": float(min_terminal_fraction),
        "max_terminal_fraction": float(max_terminal_fraction),
        "terminal_fraction_power": float(terminal_fraction_power),
        "terminal_fatea_fraction": float(terminal_fatea_fraction),
        "mass_linear_slope": float(mass_linear_slope),
        "mass_transition_boost": float(mass_transition_boost),
        "mass_wave_amp": float(mass_wave_amp),
        "mass_wave_period": float(mass_wave_period),
        "mass_wave_phase": float(mass_wave_phase),
        "mass_dip_amp": float(mass_dip_amp),
        "mass_dip_time": None if mass_dip_time is None else float(mass_dip_time),
        "mass_dip_sigma": float(mass_dip_sigma),
        "mass_bump_amp": float(mass_bump_amp),
        "mass_bump_time": None if mass_bump_time is None else float(mass_bump_time),
        "mass_bump_sigma": float(mass_bump_sigma),
        "growth_scale": float(growth_scale),
        "global_y_slope": float(global_y_slope),
        "spatial_mode": str(spatial_mode),
        "gene_input": str(gene_input),
        "critical_definition": "The pitchfork parameter r(t)=1.75*tanh((t-tc)/0.62) crosses zero at tc; one progenitor basin becomes two fate basins.",
        "potential": "Developmental potential decreases with time/commitment; weak pitchfork, spatial, and transition-rate terms shape local critical dynamics.",
        "lineage_prior": "Early Prog cells carry hidden lineage_FateA/FateB labels in obs['lineage_prior']; obs['Annotation'] remains the visible cell type.",
    }
    return adata


def plot_preview(adata: ad.AnnData, out_dir: Path, critical_time: float) -> Path:
    times = sorted(float(t) for t in adata.obs["time"].unique())
    xy_all = adata.obsm["spatial"]
    xmin, ymin = xy_all.min(axis=0)
    xmax, ymax = xy_all.max(axis=0)
    pad = 0.08 * max(xmax - xmin, ymax - ymin)
    u = adata.obs["true_potential"].to_numpy(float)
    u_min = float(np.quantile(u, 0.01))
    u_max = float(np.quantile(u, 0.99))

    fig, axes = plt.subplots(3, len(times), figsize=(2.6 * len(times), 7.2), constrained_layout=False)
    for col, time in enumerate(times):
        mask = np.isclose(adata.obs["time"].astype(float), time)
        xy = adata.obsm["spatial"][mask]
        labels = adata.obs.loc[mask, "Annotation"].astype(str).to_numpy()
        pot = adata.obs.loc[mask, "true_potential"].to_numpy(float)
        vel = adata.obsm["true_spatial_velocity"][mask]

        ax = axes[0, col]
        for cell_type in CELL_TYPES:
            m = labels == cell_type
            if np.any(m):
                ax.scatter(xy[m, 0], xy[m, 1], s=2.8, c=COLORS[cell_type], alpha=0.86, linewidths=0, rasterized=True)
        ax.set_title(f"t={time:g} type", fontsize=8)

        ax = axes[1, col]
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=2.8, c=pot, cmap="magma", vmin=u_min, vmax=u_max, alpha=0.88, linewidths=0, rasterized=True)
        ax.set_title(f"t={time:g} U", fontsize=8)

        ax = axes[2, col]
        ax.scatter(xy[:, 0], xy[:, 1], s=1.1, c="#D0D0D0", alpha=0.42, linewidths=0, rasterized=True)
        stride = max(1, int(xy.shape[0] / 95))
        idx = np.arange(xy.shape[0])[::stride]
        ax.quiver(
            xy[idx, 0],
            xy[idx, 1],
            vel[idx, 0],
            vel[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=5.6,
            width=0.003,
            color="#202020",
            alpha=0.80,
        )
        ax.set_title(f"t={time:g} velocity", fontsize=8)

        for row in range(3):
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xlim(xmin - pad, xmax + pad)
            axes[row, col].set_ylim(ymin - pad, ymax + pad)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if abs(float(time) - critical_time) < 1e-8:
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.7)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[c], markersize=5, label=c) for c in CELL_TYPES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.025, right=0.94, top=0.92, bottom=0.08, wspace=0.04, hspace=0.20)
    cax = fig.add_axes([0.955, 0.42, 0.010, 0.22])
    fig.colorbar(sc, cax=cax, label="true U")
    out = out_dir / "pitchfork_fate_critical_preview.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metrics(adata: ad.AnnData, out_dir: Path, critical_time: float) -> tuple[Path, Path]:
    obs = adata.obs.copy()
    obs["velocity_norm"] = np.linalg.norm(adata.obsm["true_spatial_velocity"], axis=1)
    rows = []
    for time, sub in obs.groupby("time", sort=True):
        counts = sub["Annotation"].astype(str).value_counts()
        rows.append(
            {
                "time": float(time),
                "total_cells": int(sub.shape[0]),
                "prog_count": int(counts.get("Prog", 0)),
                "fatea_count": int(counts.get("FateA", 0)),
                "fateb_count": int(counts.get("FateB", 0)),
                "terminal_fraction": float((sub["Annotation"].astype(str) != "Prog").mean()),
                "pitchfork_r": float(sub["pitchfork_r"].median()),
                "median_U": float(sub["true_potential"].median()),
                "q90_U": float(sub["true_potential"].quantile(0.90)),
                "median_grad_abs": float(sub["true_potential_grad_b_abs"].median()),
                "median_critical_score": float(sub["critical_score"].median()),
                "median_velocity": float(sub["velocity_norm"].median()),
                "mean_growth": float(sub["true_growth_rate"].mean()),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["step_abs_delta_q90_U"] = metrics["q90_U"].diff().abs().fillna(0.0)
    csv_path = out_dir / "ground_truth_pitchfork_critical_metrics.csv"
    metrics.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 6, figsize=(18.0, 3.0), constrained_layout=True)
    plots = [
        ("pitchfork_r", "pitchfork r", "#4C78A8"),
        ("terminal_fraction", "terminal fraction", "#E15759"),
        ("step_abs_delta_q90_U", "|step delta q90 U|", "#7B3294"),
        ("median_critical_score", "critical score", "#F28E2B"),
        ("median_velocity", "median velocity", "#202020"),
        ("mean_growth", "mean growth", "#1B9E77"),
    ]
    for ax, (key, title, color) in zip(axes, plots):
        ax.plot(metrics["time"], metrics[key], marker="o", color=color, linewidth=2)
        ax.axvline(critical_time, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    png_path = out_dir / "ground_truth_pitchfork_critical_metrics.png"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def plot_gene_pca_preview(adata: ad.AnnData, out_dir: Path) -> Path:
    times = sorted(float(t) for t in adata.obs["time"].unique())
    gene = np.asarray(adata.obsm["X_gene_input"], dtype=np.float32)
    xy_all = PCA(n_components=2, random_state=0).fit_transform(gene)
    xmin, ymin = xy_all.min(axis=0)
    xmax, ymax = xy_all.max(axis=0)
    pad = 0.08 * max(float(xmax - xmin), float(ymax - ymin), 1e-8)

    fig, axes = plt.subplots(1, len(times), figsize=(2.7 * len(times), 2.9), squeeze=False)
    for ax, time in zip(axes.ravel(), times):
        mask = np.isclose(adata.obs["time"].astype(float), time)
        labels = adata.obs.loc[mask, "Annotation"].astype(str).to_numpy()
        xy = xy_all[mask]
        for cell_type in CELL_TYPES:
            m = labels == cell_type
            if np.any(m):
                ax.scatter(xy[m, 0], xy[m, 1], s=5.0, c=COLORS[cell_type], alpha=0.82, linewidths=0, rasterized=True)
        ax.set_title(f"t={time:g}", fontsize=8)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[c], markersize=5, label=c) for c in CELL_TYPES]
    fig.suptitle("Observed gene input PCA", fontsize=12)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 0.88))
    out = out_dir / "pitchfork_fate_gene_input_pca_preview.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_count_preview(adata: ad.AnnData, out_dir: Path) -> Path:
    rows = []
    for time, sub in adata.obs.groupby("time", sort=True):
        counts = sub["Annotation"].astype(str).value_counts()
        row = {"time": float(time), "total": int(sub.shape[0])}
        for cell_type in CELL_TYPES:
            row[cell_type] = int(counts.get(cell_type, 0))
        rows.append(row)
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x_offsets = {"Prog": 0.0, "FateA": -0.035, "FateB": 0.035}
    line_styles = {"Prog": "-", "FateA": "-", "FateB": "--"}
    markers = {"Prog": "o", "FateA": "s", "FateB": "^"}
    for cell_type in CELL_TYPES:
        ax.plot(
            df["time"] + x_offsets.get(cell_type, 0.0),
            df[cell_type],
            marker=markers.get(cell_type, "o"),
            linestyle=line_styles.get(cell_type, "-"),
            linewidth=2,
            color=COLORS[cell_type],
            label=cell_type,
            alpha=0.92,
        )
    ax.plot(df["time"], df["total"], marker="x", linestyle="--", linewidth=1.8, color="#303030", label="total")
    ax.set_title("Observed cell count changes")
    ax.set_xlabel("time")
    ax.set_ylabel("cell count")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    out = out_dir / "pitchfork_fate_count_preview.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def write_dense_ground_truth(
    out_dir: Path,
    seed: int,
    critical_time: float,
    onset_time: float,
    tau: float,
    max_terminal_fraction: float,
    min_terminal_fraction: float,
    terminal_fraction_power: float,
    terminal_fatea_fraction: float,
    growth_scale: float,
    global_y_slope: float,
    spatial_mode: str,
    gt_cells: int,
    dt: float,
) -> dict[str, str]:
    times = np.round(np.arange(0.0, 4.0 + 1e-9, dt), 8).astype(float).tolist()
    exp_series: list[np.ndarray] = []
    spa_series: list[np.ndarray] = []
    label_series: list[np.ndarray] = []
    for i, time in enumerate(times):
        rng = np.random.default_rng(seed + 10000 + i)
        obs, spatial, _velocity, _growth, branch, hidden = sample_stage(
            rng,
            time,
            gt_cells,
            critical_time,
            onset_time,
            tau,
            min_terminal_fraction,
            max_terminal_fraction,
            terminal_fraction_power,
            terminal_fatea_fraction,
            growth_scale,
            global_y_slope,
            spatial_mode,
        )
        latent = build_latent(rng, obs, spatial, branch, hidden, critical_time)
        exp_series.append(latent.astype(np.float32))
        spa_series.append(spatial.astype(np.float32))
        label_series.append(obs["Annotation"].astype(str).to_numpy())

    paths = {
        "gt_exp_pkl": str(out_dir / "pitchfork_fate_critical_exp.pkl"),
        "gt_spa_pkl": str(out_dir / "pitchfork_fate_critical_spa.pkl"),
        "gt_time_pkl": str(out_dir / "pitchfork_fate_critical_time.pkl"),
        "gt_label_pkl": str(out_dir / "pitchfork_fate_critical_labels.pkl"),
    }
    with open(paths["gt_exp_pkl"], "wb") as f:
        pickle.dump(exp_series, f)
    with open(paths["gt_spa_pkl"], "wb") as f:
        pickle.dump(spa_series, f)
    with open(paths["gt_time_pkl"], "wb") as f:
        pickle.dump(times, f)
    with open(paths["gt_label_pkl"], "wb") as f:
        pickle.dump(label_series, f)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a clean pitchfork-fate ST simulation with a known critical point.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pitchfork_fate_critical_simulated"))
    parser.add_argument("--times", default=",".join(str(t) for t in DEFAULT_TIMES))
    parser.add_argument("--base-cells", type=int, default=1100)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--critical-time", type=float, default=2.0)
    parser.add_argument("--onset-time", type=float, default=0.75)
    parser.add_argument("--transition-tau", type=float, default=0.62)
    parser.add_argument("--min-terminal-fraction", type=float, default=0.0)
    parser.add_argument("--max-terminal-fraction", type=float, default=0.82)
    parser.add_argument("--terminal-fraction-power", type=float, default=1.0)
    parser.add_argument("--terminal-fatea-fraction", type=float, default=0.5)
    parser.add_argument("--mass-linear-slope", type=float, default=0.02)
    parser.add_argument("--mass-transition-boost", type=float, default=0.05)
    parser.add_argument("--mass-wave-amp", type=float, default=0.0)
    parser.add_argument("--mass-wave-period", type=float, default=2.8)
    parser.add_argument("--mass-wave-phase", type=float, default=0.0)
    parser.add_argument("--mass-dip-amp", type=float, default=0.0)
    parser.add_argument("--mass-dip-time", type=float, default=None)
    parser.add_argument("--mass-dip-sigma", type=float, default=0.5)
    parser.add_argument("--mass-bump-amp", type=float, default=0.0)
    parser.add_argument("--mass-bump-time", type=float, default=None)
    parser.add_argument("--mass-bump-sigma", type=float, default=0.5)
    parser.add_argument("--growth-scale", type=float, default=0.30)
    parser.add_argument(
        "--spatial-mode",
        choices=["gaussian", "regular"],
        default="gaussian",
        help="Use 'regular' for a compact smooth tissue template with gradual pitchfork branches.",
    )
    parser.add_argument(
        "--global-y-slope",
        type=float,
        default=0.0,
        help="Global vertical tissue drift per time unit. Use 0 to avoid whole-slice upward translation.",
    )
    parser.add_argument("--gene-input", choices=["true_latent", "pca"], default="true_latent")
    parser.add_argument("--gt-dt", type=float, default=0.05)
    parser.add_argument("--gt-cells", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times = [float(x.strip()) for x in str(args.times).split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = build_adata(
        rng=rng,
        times=times,
        base_cells=args.base_cells,
        n_genes=args.n_genes,
        critical_time=args.critical_time,
        onset_time=args.onset_time,
        tau=args.transition_tau,
        min_terminal_fraction=args.min_terminal_fraction,
        max_terminal_fraction=args.max_terminal_fraction,
        terminal_fraction_power=args.terminal_fraction_power,
        terminal_fatea_fraction=args.terminal_fatea_fraction,
        gene_input=args.gene_input,
        mass_linear_slope=args.mass_linear_slope,
        mass_transition_boost=args.mass_transition_boost,
        mass_wave_amp=args.mass_wave_amp,
        mass_wave_period=args.mass_wave_period,
        mass_wave_phase=args.mass_wave_phase,
        mass_dip_amp=args.mass_dip_amp,
        mass_dip_time=args.mass_dip_time,
        mass_dip_sigma=args.mass_dip_sigma,
        mass_bump_amp=args.mass_bump_amp,
        mass_bump_time=args.mass_bump_time,
        mass_bump_sigma=args.mass_bump_sigma,
        growth_scale=args.growth_scale,
        global_y_slope=args.global_y_slope,
        spatial_mode=args.spatial_mode,
    )
    h5ad_path = args.output_dir / "pitchfork_fate_critical_stvcr_ready.h5ad"
    adata.write_h5ad(h5ad_path)
    preview_path = plot_preview(adata, args.output_dir, args.critical_time)
    gene_pca_preview = plot_gene_pca_preview(adata, args.output_dir)
    count_preview = plot_count_preview(adata, args.output_dir)
    metrics_csv, metrics_png = plot_metrics(adata, args.output_dir, args.critical_time)
    gt_paths = write_dense_ground_truth(
        out_dir=args.output_dir,
        seed=args.seed,
        critical_time=args.critical_time,
        onset_time=args.onset_time,
        tau=args.transition_tau,
        min_terminal_fraction=args.min_terminal_fraction,
        max_terminal_fraction=args.max_terminal_fraction,
        terminal_fraction_power=args.terminal_fraction_power,
        terminal_fatea_fraction=args.terminal_fatea_fraction,
        growth_scale=args.growth_scale,
        global_y_slope=args.global_y_slope,
        spatial_mode=args.spatial_mode,
        gt_cells=args.gt_cells,
        dt=args.gt_dt,
    )

    summary = {
        "h5ad": str(h5ad_path),
        "preview_png": str(preview_path),
        "gene_pca_preview_png": str(gene_pca_preview),
        "count_preview_png": str(count_preview),
        "critical_metrics_csv": str(metrics_csv),
        "critical_metrics_png": str(metrics_png),
        **gt_paths,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "times": times,
        "cell_types": CELL_TYPES,
        "ground_truth_critical_time": float(args.critical_time),
        "gene_input": str(args.gene_input),
        "min_terminal_fraction": float(args.min_terminal_fraction),
        "terminal_fraction_power": float(args.terminal_fraction_power),
        "terminal_fatea_fraction": float(args.terminal_fatea_fraction),
        "mass_linear_slope": float(args.mass_linear_slope),
        "mass_transition_boost": float(args.mass_transition_boost),
        "mass_wave_amp": float(args.mass_wave_amp),
        "mass_wave_period": float(args.mass_wave_period),
        "mass_wave_phase": float(args.mass_wave_phase),
        "mass_dip_amp": float(args.mass_dip_amp),
        "mass_dip_time": None if args.mass_dip_time is None else float(args.mass_dip_time),
        "mass_dip_sigma": float(args.mass_dip_sigma),
        "mass_bump_amp": float(args.mass_bump_amp),
        "mass_bump_time": None if args.mass_bump_time is None else float(args.mass_bump_time),
        "mass_bump_sigma": float(args.mass_bump_sigma),
        "growth_scale": float(args.growth_scale),
        "global_y_slope": float(args.global_y_slope),
        "spatial_mode": str(args.spatial_mode),
        "annotation_counts_by_time": {
            str(time): adata.obs.loc[np.isclose(adata.obs["time"].astype(float), time), "Annotation"].astype(str).value_counts().to_dict()
            for time in times
        },
        "keys": {
            "annotation": "obs['Annotation']",
            "lineage_prior": "obs['lineage_prior']",
            "terminal_fate": "obs['terminal_fate']",
            "time": "obs['time']",
            "time_input": "obs['time_input']",
            "spatial": "obsm['spatial']",
            "model_spatial": "obsm['X_spatial_input']",
            "model_gene": "obsm['X_gene_input']",
            "pca_gene": "obsm['X_pca']",
            "true_latent": "obsm['X_true_latent']",
            "true_velocity": "obsm['true_spatial_velocity']",
            "true_potential": "obs['true_potential']",
            "true_growth": "obs['true_growth_rate']",
            "pitchfork_r": "obs['pitchfork_r']",
            "critical_score": "obs['critical_score']",
        },
    }
    summary_path = args.output_dir / "pitchfork_fate_critical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme_path = args.output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Pitchfork Fate Critical Simulation",
                "",
                "This is a synthetic spatial transcriptomics benchmark with a known critical point.",
                "",
                f"- Critical time: `{args.critical_time}`",
                f"- Total mass is weak and can be nonmonotonic: `mass_linear_slope={args.mass_linear_slope}`, `mass_transition_boost={args.mass_transition_boost}`, `mass_wave_amp={args.mass_wave_amp}`.",
                f"- Terminal fraction opens gradually with power `{args.terminal_fraction_power}`.",
                f"- Initial terminal fraction is `{args.min_terminal_fraction}`.",
                f"- Cell-level true growth is scaled by `{args.growth_scale}`.",
                f"- Spatial mode: `{args.spatial_mode}`.",
                f"- Global y drift is controlled by `global_y_slope={args.global_y_slope}`; `0` keeps the whole tissue from translating upward.",
                "- Visible cell types: `Prog`, `FateA`, `FateB`",
                "- Early `Prog` cells carry hidden future fate in `obs['lineage_prior']`.",
                "- The true potential is `U(b,t)=0.25*b^4-0.5*r(t)*b^2` plus weak spatial and transition-rate terms.",
                "- `r(t)` crosses zero at the critical time, making one progenitor basin split into two terminal basins.",
                "- `obsm['X_gene_input']` is controlled by `--gene-input`; default is clean 10D true latent for dynamics debugging.",
                "",
                "Main files:",
                f"- `{h5ad_path.name}`",
                f"- `{preview_path.name}`",
                f"- `{gene_pca_preview.name}`",
                f"- `{count_preview.name}`",
                f"- `{metrics_csv.name}`",
                f"- `{metrics_png.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

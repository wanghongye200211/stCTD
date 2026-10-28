#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.interpolate import griddata
from scipy.spatial import cKDTree, distance_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERTURBATION_SCRIPT_DIR = PROJECT_ROOT / "psg_full_ruot_gat" / "scripts"
if str(PERTURBATION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PERTURBATION_SCRIPT_DIR))

import run_dorsal_gene_critical_perturbation as perturb  # noqa: E402


ACTIVE_RUN = PROJECT_ROOT / "SpaPOT_runs" / "dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign"
DEFAULT_PHATE_CSV = (
    PROJECT_ROOT
    / "result_figures"
    / "dorsal_midbrain_12p5_14p5_16p5_hybrid"
    / "spatial_reconstruction_story_v2"
    / "fig04_observed_expression_potential_embeddings_v1"
    / "observed_expression_pca50_phate_source.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "result_figures" / "小鼠中脑" / "g扰动"
DEFAULT_GAT_CHECKPOINT = ACTIVE_RUN / "gene_prior_gatae_retrain_20260523" / "gene_prior_gatae_10d.pt"

CELL_COLORS = {"RGC": "#2ca25f", "NeuB": "#f28e2b", "GlioB": "#4e79a7"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw paired baseline and gene-perturbed trajectories for the same E12.5 RGC cells, "
            "projected onto the observed PHATE potential landscape."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=ACTIVE_RUN)
    parser.add_argument("--phate-csv", type=Path, default=DEFAULT_PHATE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_GAT_CHECKPOINT)
    parser.add_argument("--embedding", choices=["phate", "umap"], default="phate")
    parser.add_argument("--background-style", choices=["potential", "points", "density", "none"], default="potential")
    parser.add_argument("--umap-n-neighbors", type=int, default=35)
    parser.add_argument("--umap-min-dist", type=float, default=0.25)
    parser.add_argument("--umap-spread", type=float, default=1.0)
    parser.add_argument("--gene", default="Efna5")
    parser.add_argument("--dose", type=float, default=4.0)
    parser.add_argument("--source-time", type=float, default=12.5)
    parser.add_argument("--terminal-time", type=float, default=16.5)
    parser.add_argument("--source-label", default="RGC")
    parser.add_argument("--from-fate", default="NeuB")
    parser.add_argument("--to-fate", default="GlioB")
    parser.add_argument("--max-source-cells", type=int, default=384)
    parser.add_argument("--plot-cells", type=int, default=28)
    parser.add_argument("--seed", type=int, default=19491001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--metric-dt", type=float, default=0.5)
    parser.add_argument("--ode-chunk-size", type=int, default=256)
    parser.add_argument("--gat-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=2048)
    parser.add_argument("--knn-k", type=int, default=31)
    parser.add_argument("--projection-k", type=int, default=20)
    parser.add_argument("--expression-clip-quantile", type=float, default=0.999)
    parser.add_argument("--latent-add-x", type=float, default=0.32)
    parser.add_argument("--latent-add-max-x", type=float, default=0.64)
    parser.add_argument("--latent-direction-expression-dose", type=float, default=1.0)
    parser.add_argument("--grid", type=int, default=180)
    parser.add_argument("--fig-width", type=float, default=4.25)
    parser.add_argument("--fig-height", type=float, default=3.65)
    parser.add_argument("--title-mode", choices=["none", "compact"], default="none")
    parser.add_argument("--legend-mode", choices=["none", "direct", "legend"], default="legend")
    parser.add_argument("--control-path-mode", choices=["rollout", "straight"], default="rollout")
    parser.add_argument("--anchor-perturbed-start", action="store_true")
    parser.add_argument("--perturbed-line-style", choices=["solid", "dashed"], default="solid")
    parser.add_argument("--endpoint-link-alpha", type=float, default=0.0)
    parser.add_argument(
        "--cell-selection",
        choices=["probability_gain", "compact_perturbed_phate", "compact_both_phate"],
        default="probability_gain",
    )
    parser.add_argument("--compact-pool-size", type=int, default=96)
    parser.add_argument("--endpoint-emphasis", action="store_true")
    parser.add_argument("--endpoint-labels", choices=["none", "badge"], default="none")
    parser.add_argument("--endpoint-zone-labels", choices=["none", "direct"], default="none")
    parser.add_argument("--endpoint-zone-alpha", type=float, default=0.16)
    parser.add_argument("--endpoint-marker-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-callout", choices=["none", "split", "strip"], default="none")
    parser.add_argument("--endpoint-display", choices=["all", "centroid"], default="all")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, stem: Path, dpi: int = 600) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg", "tiff"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs[suffix] = str(path)
    plt.close(fig)
    return outputs


def dense_float32(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        max_source_cells=int(args.max_source_cells),
        source_label=str(args.source_label),
        metric_dt=float(args.metric_dt),
        ode_chunk_size=int(args.ode_chunk_size),
        gat_batch_size=int(args.gat_batch_size),
        classifier_batch_size=int(args.classifier_batch_size),
        knn_k=int(args.knn_k),
        expression_clip_quantile=float(args.expression_clip_quantile),
        perturbation_mode="latent_add",
        latent_add_x=float(args.latent_add_x),
        latent_add_max_x=float(args.latent_add_max_x),
        latent_direction_expression_dose=float(args.latent_direction_expression_dose),
        ot_sample_size=128,
        sinkhorn_iters=30,
        sinkhorn_epsilon_scale=0.08,
        gene_input_key="X_gene_input",
        spatial_key="X_spatial_aligned",
        time_input_key="time_input",
        raw_time_key="time",
        annotation_key="Annotation",
        decoder_checkpoint=None,
    )


def record_rollout(
    *,
    model: Any,
    state0: torch.Tensor,
    logw0: torch.Tensor,
    source_input: float,
    terminal_input: float,
    method: str,
    step_size: float,
    metric_dt: float,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = max(1, int(np.ceil((float(terminal_input) - float(source_input)) / max(float(metric_dt), 1e-8))))
    times = np.linspace(float(source_input), float(terminal_input), n_steps + 1)
    states = [state0.detach().cpu().numpy().astype(np.float32)]
    state = state0.detach()
    logw = logw0.detach()
    for idx in range(1, len(times)):
        state, logw = perturb.integrate_model(
            model,
            state,
            logw,
            float(times[idx - 1]),
            float(times[idx]),
            method=method,
            step_size=step_size,
            chunk_size=chunk_size,
            device=device,
        )
        states.append(state.detach().cpu().numpy().astype(np.float32))
    return times.astype(np.float32), np.stack(states, axis=0)


def load_phate_projector(
    adata: ad.AnnData,
    phate_csv: Path,
    latent_key: str,
    projection_k: int,
) -> tuple[pd.DataFrame, cKDTree, np.ndarray, np.ndarray, np.ndarray, int]:
    phate = pd.read_csv(phate_csv.expanduser().resolve())
    if "label" not in phate.columns and "cell_type" in phate.columns:
        phate["label"] = phate["cell_type"].astype(str)
    obs_to_idx = {str(name): idx for idx, name in enumerate(adata.obs_names)}
    if "adata_obs_name" in phate.columns:
        idx = np.asarray([obs_to_idx.get(str(name), -1) for name in phate["adata_obs_name"]], dtype=int)
        keep = idx >= 0
        phate = phate.loc[keep].reset_index(drop=True)
        idx = idx[keep]
    else:
        idx = np.arange(min(adata.n_obs, len(phate)), dtype=int)
        phate = phate.iloc[: len(idx)].reset_index(drop=True)
    latent = np.asarray(adata.obsm[latent_key][idx], dtype=np.float32)
    mean = latent.mean(axis=0, keepdims=True)
    std = latent.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    tree = cKDTree(((latent - mean) / std).astype(np.float32))
    k = int(min(max(1, projection_k), latent.shape[0]))
    return phate, tree, phate[["pca50_phate1", "pca50_phate2"]].to_numpy(np.float32), mean, std, k


def load_state_projector(
    adata: ad.AnnData,
    phate_csv: Path,
    latent_key: str,
    projection_k: int,
    *,
    embedding: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_spread: float,
    seed: int,
) -> tuple[pd.DataFrame, cKDTree, np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    phate, _tree, _xy, _mean, _std, _k = load_phate_projector(adata, phate_csv, latent_key, projection_k)
    obs_to_idx = {str(name): idx for idx, name in enumerate(adata.obs_names)}
    idx = np.asarray([obs_to_idx[str(name)] for name in phate["adata_obs_name"]], dtype=int)
    latent = np.asarray(adata.obsm[latent_key][idx], dtype=np.float32)
    mean = latent.mean(axis=0, keepdims=True)
    std = latent.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    latent_scaled = ((latent - mean) / std).astype(np.float32)
    k = int(min(max(1, projection_k), latent.shape[0]))
    tree = cKDTree(latent_scaled)

    info: dict[str, Any] = {"embedding": str(embedding), "latent_key": str(latent_key)}
    if str(embedding) == "phate":
        plot_df = phate.copy()
        plot_df["embed1"] = plot_df["pca50_phate1"].astype(float)
        plot_df["embed2"] = plot_df["pca50_phate2"].astype(float)
        ref_xy = plot_df[["embed1", "embed2"]].to_numpy(np.float32)
        info.update({"source": str(phate_csv), "coordinates": "pca50_phate"})
        return plot_df, tree, ref_xy, mean, std, k, info

    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=int(umap_n_neighbors),
        min_dist=float(umap_min_dist),
        spread=float(umap_spread),
        metric="euclidean",
        random_state=int(seed),
        transform_seed=int(seed),
        low_memory=True,
    )
    xy = reducer.fit_transform(latent_scaled).astype(np.float32)
    plot_df = phate.copy()
    plot_df["embed1"] = xy[:, 0]
    plot_df["embed2"] = xy[:, 1]
    info.update(
        {
            "fit": "UMAP on standardized current model gene-state latent for observed cells",
            "n_neighbors": int(umap_n_neighbors),
            "min_dist": float(umap_min_dist),
            "spread": float(umap_spread),
        }
    )
    return plot_df, tree, xy, mean, std, k, info


def project_states_to_phate(
    states: np.ndarray,
    *,
    spatial_dim: int,
    tree: cKDTree,
    ref_xy: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    k: int,
) -> np.ndarray:
    shape = states.shape[:-1]
    latent = states.reshape(-1, states.shape[-1])[:, spatial_dim:].astype(np.float32)
    query = ((latent - mean) / std).astype(np.float32)
    dist, idx = tree.query(query, k=k)
    dist = np.asarray(dist, dtype=np.float32)
    idx = np.asarray(idx, dtype=int)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    scale = np.nanmedian(dist[:, -1])
    scale = max(float(scale), 1e-6)
    weights = np.exp(-(dist**2) / (2.0 * scale * scale)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    xy = (weights[..., None] * ref_xy[idx]).sum(axis=1)
    return xy.reshape(*shape, 2).astype(np.float32)


def draw_background(ax: plt.Axes, embedding_df: pd.DataFrame, grid: int, *, style: str) -> None:
    x = embedding_df["embed1"].to_numpy(float)
    y = embedding_df["embed2"].to_numpy(float)
    xpad = 0.035 * max(float(x.max() - x.min()), 1e-8)
    ypad = 0.035 * max(float(y.max() - y.min()), 1e-8)
    xlo, xhi = float(x.min() - xpad), float(x.max() + xpad)
    ylo, yhi = float(y.min() - ypad), float(y.max() + ypad)
    if str(style) == "potential":
        u = embedding_df["potential_U"].to_numpy(float)
        xi = np.linspace(xlo, xhi, int(grid))
        yi = np.linspace(ylo, yhi, int(grid))
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata(np.column_stack([x, y]), u, (xx, yy), method="linear")
        levels = np.linspace(np.nanpercentile(u, 3), np.nanpercentile(u, 97), 16)
        ax.contourf(xx, yy, zz, levels=levels, cmap="Greys", alpha=0.34, antialiased=True, zorder=0)
    elif str(style) == "density":
        ax.hexbin(x, y, gridsize=52, cmap="Greys", mincnt=1, linewidths=0, alpha=0.13, zorder=0)
    if str(style) != "none":
        labels = embedding_df["label"].astype(str).to_numpy()
        for label, color in CELL_COLORS.items():
            mask = labels == label
            alpha = 0.18 if str(style) in {"points", "density"} else 0.15
            size = 3.2 if str(style) in {"points", "density"} else 3.0
            ax.scatter(x[mask], y[mask], s=size, c=color, alpha=alpha, linewidths=0, rasterized=True, zorder=1)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def line_with_shadow(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    color: str,
    lw: float,
    alpha: float,
    zorder: int,
    linestyle: str = "solid",
) -> None:
    ls = (0, (3.2, 2.2)) if linestyle == "dashed" else "solid"
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=7.5, alpha=0.075, linestyle=ls, solid_capstyle="round", zorder=zorder)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, linestyle=ls, solid_capstyle="round", zorder=zorder + 1)


def straightened_paths(xy: np.ndarray) -> np.ndarray:
    frac = np.linspace(0.0, 1.0, xy.shape[0], dtype=np.float32)[:, None, None]
    return ((1.0 - frac) * xy[0:1, :, :] + frac * xy[-1:, :, :]).astype(np.float32)


def arrow_on_curve(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    color: str,
    frac: float,
    mutation_scale: float,
    lw: float,
    zorder: int,
) -> None:
    if xy.shape[0] < 2:
        return
    i0 = int(np.clip(round((xy.shape[0] - 2) * frac), 0, xy.shape[0] - 2))
    p0 = xy[i0]
    p1 = xy[i0 + 1]
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
        return
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        lw=lw,
        color=color,
        alpha=0.98,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        path_effects=[pe.withStroke(linewidth=lw + 1.8, foreground="white", alpha=0.78)],
    )
    ax.add_patch(patch)


def direct_label(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    color: str,
    ha: str = "left",
    va: str = "center",
) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        ha=ha,
        va=va,
        fontsize=6.8,
        weight="bold",
        color=color,
        clip_on=True,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.96)],
        zorder=30,
    )


def choose_cells(
    base_labels: np.ndarray,
    pert_labels: np.ndarray,
    base_prob_to: np.ndarray,
    pert_prob_to: np.ndarray,
    from_fate: str,
    to_fate: str,
    n_plot: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    strict = np.flatnonzero((base_labels == from_fate) & (pert_labels == to_fate))
    mode = f"{from_fate}->{to_fate}"
    if len(strict) >= max(3, min(8, int(n_plot))):
        score = pert_prob_to[strict] - base_prob_to[strict]
        order = strict[np.argsort(score)[::-1]]
        return order[: int(n_plot)], mode
    relaxed = np.flatnonzero(base_labels == from_fate)
    mode = f"{from_fate}-bound; ranked by {to_fate} gain"
    if len(relaxed) == 0:
        relaxed = np.arange(len(base_labels))
        mode = f"all source cells; ranked by {to_fate} gain"
    score = pert_prob_to[relaxed] - base_prob_to[relaxed]
    order = relaxed[np.argsort(score)[::-1]]
    if len(order) > int(n_plot):
        return order[: int(n_plot)], mode
    rng = np.random.default_rng(int(seed))
    return rng.choice(order, size=min(len(order), int(n_plot)), replace=False), mode


def compact_subset(indices: np.ndarray, terminal_xy: np.ndarray, n_plot: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= int(n_plot):
        return indices
    xy = np.asarray(terminal_xy[indices], dtype=np.float32)
    finite = np.all(np.isfinite(xy), axis=1)
    if int(np.sum(finite)) < int(n_plot):
        return indices[: int(n_plot)]
    indices = indices[finite]
    xy = xy[finite]
    d = distance_matrix(xy, xy)
    k = min(int(n_plot), len(indices))
    score = np.sort(d, axis=1)[:, :k].mean(axis=1)
    center_pos = int(np.argmin(score))
    nearest = np.argsort(d[center_pos])[:k]
    return indices[nearest]


def compact_both_endpoint_subset(
    indices: np.ndarray,
    base_terminal_xy: np.ndarray,
    pert_terminal_xy: np.ndarray,
    n_plot: int,
    gain: np.ndarray | None = None,
) -> np.ndarray:
    """Choose cells whose control and perturbed endpoints are both spatially coherent."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= int(n_plot):
        return indices
    base_xy = np.asarray(base_terminal_xy[indices], dtype=np.float32)
    pert_xy = np.asarray(pert_terminal_xy[indices], dtype=np.float32)
    finite = np.all(np.isfinite(base_xy), axis=1) & np.all(np.isfinite(pert_xy), axis=1)
    if int(np.sum(finite)) < int(n_plot):
        return indices[: int(n_plot)]
    indices = indices[finite]
    base_xy = base_xy[finite]
    pert_xy = pert_xy[finite]
    gain_arr: np.ndarray | None = None
    if gain is not None:
        gain_arr = np.asarray(gain, dtype=np.float32)[finite]

    # Put the two endpoint spaces on the same robust distance scale before combining them.
    base_scale = np.nanpercentile(distance_matrix(base_xy, base_xy), 75)
    pert_scale = np.nanpercentile(distance_matrix(pert_xy, pert_xy), 75)
    base_scale = max(float(base_scale), 1e-6)
    pert_scale = max(float(pert_scale), 1e-6)
    base_d = distance_matrix(base_xy, base_xy) / base_scale
    pert_d = distance_matrix(pert_xy, pert_xy) / pert_scale
    both_d = 0.42 * base_d + 0.58 * pert_d
    k = min(int(n_plot), len(indices))
    shift = np.linalg.norm(pert_xy - base_xy, axis=1)
    best_score = np.inf
    selected_pos = np.argsort(both_d[0])[:k]
    for center_pos in range(len(indices)):
        candidate_pos = np.argsort(both_d[center_pos])[:k]
        base_to_center = base_d[center_pos, candidate_pos]
        pert_to_center = pert_d[center_pos, candidate_pos]
        compact_score = (
            0.35 * float(np.mean(base_to_center))
            + 0.20 * float(np.quantile(base_to_center, 0.85))
            + 0.45 * float(np.mean(pert_to_center))
            + 0.28 * float(np.quantile(pert_to_center, 0.85))
        )
        shift_bonus = float(np.nanmedian(shift[candidate_pos])) / max(float(np.nanpercentile(shift, 75)), 1e-6)
        gain_bonus = 0.0
        if gain_arr is not None and np.nanstd(gain_arr) > 1e-8:
            gain_z = (gain_arr - np.nanmedian(gain_arr)) / (np.nanstd(gain_arr) + 1e-8)
            gain_bonus = float(np.nanmean(gain_z[candidate_pos]))
        score = compact_score - 0.08 * shift_bonus - 0.045 * gain_bonus
        if score < best_score:
            best_score = score
            selected_pos = candidate_pos
    return indices[selected_pos]


def choose_cells_with_optional_compaction(
    *,
    args: argparse.Namespace,
    base_labels: np.ndarray,
    pert_labels: np.ndarray,
    base_probs: np.ndarray,
    pert_probs: np.ndarray,
    to_idx: int,
    base_terminal_states: np.ndarray,
    pert_terminal_states: np.ndarray,
    spatial_dim: int,
    tree: cKDTree,
    ref_xy: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    projection_k: int,
) -> tuple[np.ndarray, str]:
    from_fate = str(args.from_fate)
    to_fate = str(args.to_fate)
    n_plot = int(args.plot_cells)
    strict = np.flatnonzero((base_labels == from_fate) & (pert_labels == to_fate))
    if str(args.cell_selection) == "compact_both_phate" and len(strict) >= max(3, min(8, n_plot)):
        gain = pert_probs[strict, to_idx] - base_probs[strict, to_idx]
        pool_order = strict[np.argsort(gain)[::-1]]
        pool = pool_order[: min(len(pool_order), max(n_plot, int(args.compact_pool_size)))]
        base_terminal_xy = project_states_to_phate(
            base_terminal_states[pool],
            spatial_dim=spatial_dim,
            tree=tree,
            ref_xy=ref_xy,
            mean=ref_mean,
            std=ref_std,
            k=projection_k,
        )
        pert_terminal_xy = project_states_to_phate(
            pert_terminal_states[pool],
            spatial_dim=spatial_dim,
            tree=tree,
            ref_xy=ref_xy,
            mean=ref_mean,
            std=ref_std,
            k=projection_k,
        )
        pool_gain = pert_probs[pool, to_idx] - base_probs[pool, to_idx]
        selected_pos = compact_both_endpoint_subset(
            np.arange(len(pool), dtype=int),
            base_terminal_xy,
            pert_terminal_xy,
            n_plot,
            gain=pool_gain,
        )
        selected = pool[selected_pos]
        return selected, f"{from_fate}->{to_fate}; compact control and perturbed endpoints"
    if str(args.cell_selection) == "compact_perturbed_phate" and len(strict) >= max(3, min(8, n_plot)):
        gain = pert_probs[strict, to_idx] - base_probs[strict, to_idx]
        pool_order = strict[np.argsort(gain)[::-1]]
        pool = pool_order[: min(len(pool_order), max(n_plot, int(args.compact_pool_size)))]
        terminal_xy = project_states_to_phate(
            pert_terminal_states[pool],
            spatial_dim=spatial_dim,
            tree=tree,
            ref_xy=ref_xy,
            mean=ref_mean,
            std=ref_std,
            k=projection_k,
        )
        selected_pos = compact_subset(np.arange(len(pool), dtype=int), terminal_xy, n_plot)
        selected = pool[selected_pos]
        return selected, f"{from_fate}->{to_fate}; compact {to_fate} endpoints"
    return choose_cells(
        base_labels,
        pert_labels,
        base_probs[:, to_idx],
        pert_probs[:, to_idx],
        from_fate,
        to_fate,
        n_plot,
        int(args.seed),
    )


def add_endpoint_badges(
    ax: plt.Axes,
    base_end: np.ndarray,
    pert_end: np.ndarray,
    *,
    from_fate: str,
    to_fate: str,
    base_color: str,
    pert_color: str,
) -> None:
    for y, text, color, marker in (
        (0.895, f"circle: control end = {from_fate}", base_color, "o"),
        (0.845, f"diamond: perturbed end = {to_fate}", pert_color, "D"),
    ):
        ax.scatter(
            [0.64],
            [y],
            transform=ax.transAxes,
            s=22,
            c=color,
            marker=marker,
            edgecolors="#111111",
            linewidths=0.28,
            zorder=36,
            clip_on=False,
        )
        ax.text(
            0.665,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.4,
            weight="bold",
            color=color,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.96)],
            zorder=35,
            clip_on=False,
        )


def covariance_ellipse(
    xy: np.ndarray,
    *,
    scale: float = 2.15,
    min_width: float = 0.00055,
    min_height: float = 0.00055,
) -> tuple[tuple[float, float], float, float, float] | None:
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 3:
        return None
    center = tuple(np.median(xy, axis=0))
    cov = np.cov(xy.T)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width = max(float(2.0 * scale * np.sqrt(vals[0])), float(min_width))
    height = max(float(2.0 * scale * np.sqrt(vals[1])), float(min_height))
    return center, width, height, angle


def add_endpoint_zone(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    color: str,
    alpha: float,
    zorder: int,
) -> tuple[float, float]:
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    center = tuple(np.median(xy, axis=0)) if len(xy) else (0.0, 0.0)
    ellipse = covariance_ellipse(xy)
    if ellipse is None:
        return center
    (cx, cy), width, height, angle = ellipse
    patch = Ellipse(
        (cx, cy),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.05,
        alpha=float(alpha),
        zorder=zorder,
    )
    ax.add_patch(patch)
    outline = Ellipse(
        (cx, cy),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=0.95,
        alpha=0.82,
        zorder=zorder + 1,
    )
    ax.add_patch(outline)
    return (float(cx), float(cy))


def add_endpoint_direct_labels(
    ax: plt.Axes,
    *,
    base_center: tuple[float, float],
    pert_center: tuple[float, float],
    from_fate: str,
    to_fate: str,
    base_color: str,
    pert_color: str,
) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = 0.070 * max(float(x1 - x0), 1e-8)
    dy = 0.088 * max(float(y1 - y0), 1e-8)
    base_side = 1.0 if base_center[0] >= pert_center[0] else -1.0
    pert_side = -base_side
    entries = (
        (base_center, (base_center[0] + base_side * dx, base_center[1] + dy), f"Control end\n{from_fate}", base_color, "o"),
        (pert_center, (pert_center[0] + pert_side * dx, pert_center[1] - dy * 0.72), f"Perturbed end\n{to_fate}", pert_color, "D"),
    )
    for center, label_xy, text, color, marker in entries:
        ax.plot(
            [center[0], label_xy[0]],
            [center[1], label_xy[1]],
            color=color,
            lw=0.75,
            alpha=0.86,
            zorder=40,
            path_effects=[pe.withStroke(linewidth=2.1, foreground="white", alpha=0.90)],
        )
        ax.scatter(
            [label_xy[0]],
            [label_xy[1]],
            s=28,
            c=color,
            marker=marker,
            edgecolors="#111111",
            linewidths=0.36,
            zorder=42,
            clip_on=False,
        )
        ax.text(
            label_xy[0] + (0.014 * (x1 - x0) * (1 if label_xy[0] >= center[0] else -1)),
            label_xy[1],
            text,
            ha="left" if label_xy[0] >= center[0] else "right",
            va="center",
            fontsize=5.75,
            weight="bold",
            color=color,
            linespacing=0.88,
            clip_on=False,
            zorder=43,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.98)],
        )


def add_endpoint_centroid_labels(
    ax: plt.Axes,
    *,
    base_center: tuple[float, float],
    pert_center: tuple[float, float],
    from_fate: str,
    to_fate: str,
    base_color: str,
    pert_color: str,
) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = 0.050 * max(float(x1 - x0), 1e-8)
    dy = 0.052 * max(float(y1 - y0), 1e-8)
    base_side = 1.0 if base_center[0] >= pert_center[0] else -1.0
    pert_side = -base_side
    entries = (
        (base_center, (base_center[0] + base_side * dx, base_center[1] + dy), f"Control {from_fate}", base_color),
        (pert_center, (pert_center[0] + pert_side * dx, pert_center[1] - dy), f"Perturbed {to_fate}", pert_color),
    )
    for center, label_xy, text, color in entries:
        ax.plot(
            [center[0], label_xy[0]],
            [center[1], label_xy[1]],
            color=color,
            lw=0.56,
            alpha=0.72,
            zorder=38,
            path_effects=[pe.withStroke(linewidth=1.65, foreground="white", alpha=0.92)],
        )
        ax.text(
            label_xy[0],
            label_xy[1],
            text,
            ha="left" if label_xy[0] >= center[0] else "right",
            va="center",
            fontsize=5.55,
            weight="bold",
            color=color,
            zorder=39,
            path_effects=[pe.withStroke(linewidth=2.25, foreground="white", alpha=0.98)],
        )


def endpoint_callout_y(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    y = xy[:, 1]
    finite = np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return np.linspace(-0.32, 0.32, len(y), dtype=float)
    order = np.argsort(y)
    spread = np.linspace(-0.36, 0.36, len(y), dtype=float)
    out = np.zeros(len(y), dtype=float)
    out[order] = spread
    return out


def add_endpoint_split_callout(
    ax: plt.Axes,
    base_end: np.ndarray,
    pert_end: np.ndarray,
    *,
    from_fate: str,
    to_fate: str,
    base_color: str,
    pert_color: str,
) -> None:
    axins = ax.inset_axes([0.040, 0.055, 0.315, 0.255], zorder=55)
    axins.set_facecolor((1.0, 1.0, 1.0, 0.90))
    axins.axvspan(0.02, 0.49, color=base_color, alpha=0.075, lw=0)
    axins.axvspan(0.51, 0.98, color=pert_color, alpha=0.085, lw=0)
    axins.axvline(0.5, color="#777777", lw=0.55, alpha=0.45)

    y_base = endpoint_callout_y(base_end)
    y_pert = endpoint_callout_y(pert_end)
    x_base = np.full(len(y_base), 0.27)
    x_pert = np.full(len(y_pert), 0.73)
    if len(y_base) > 1:
        x_base += 0.026 * np.sin(np.arange(len(y_base)) * 1.9)
    if len(y_pert) > 1:
        x_pert += 0.026 * np.cos(np.arange(len(y_pert)) * 1.7)

    axins.scatter(
        x_base,
        y_base,
        s=20,
        c=base_color,
        marker="o",
        edgecolors="#111111",
        linewidths=0.35,
        alpha=0.92,
        zorder=3,
    )
    axins.scatter(
        x_pert,
        y_pert,
        s=28,
        c=pert_color,
        marker="D",
        edgecolors="#111111",
        linewidths=0.38,
        alpha=0.96,
        zorder=4,
    )
    axins.text(
        0.02,
        1.03,
        "Endpoint class",
        transform=axins.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.35,
        weight="bold",
        color="#333333",
    )
    axins.text(
        0.27,
        -0.52,
        f"Control\n{from_fate}",
        ha="center",
        va="top",
        fontsize=5.1,
        weight="bold",
        color=base_color,
        linespacing=0.86,
    )
    axins.text(
        0.73,
        -0.52,
        f"Perturbed\n{to_fate}",
        ha="center",
        va="top",
        fontsize=5.1,
        weight="bold",
        color=pert_color,
        linespacing=0.86,
    )
    axins.set_xlim(0.0, 1.0)
    axins.set_ylim(-0.56, 0.56)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#777777")


def add_endpoint_strip_callout(
    ax: plt.Axes,
    base_end: np.ndarray,
    pert_end: np.ndarray,
    *,
    from_fate: str,
    to_fate: str,
    base_color: str,
    pert_color: str,
) -> None:
    axins = ax.inset_axes([0.048, 0.072, 0.265, 0.205], zorder=55)
    axins.set_facecolor((1.0, 1.0, 1.0, 0.0))
    bg = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.018,rounding_size=0.052",
        transform=axins.transAxes,
        facecolor=(1.0, 1.0, 1.0, 0.76),
        edgecolor=(0.72, 0.72, 0.72, 0.50),
        linewidth=0.46,
        zorder=0,
        clip_on=False,
    )
    axins.add_patch(bg)
    axins.axvspan(0.065, 0.455, color=base_color, alpha=0.060, lw=0, zorder=1)
    axins.axvspan(0.545, 0.935, color=pert_color, alpha=0.065, lw=0, zorder=1)

    y_base = endpoint_callout_y(base_end) * 0.76 - 0.015
    y_pert = endpoint_callout_y(pert_end) * 0.76 - 0.015
    x_base = np.full(len(y_base), 0.265)
    x_pert = np.full(len(y_pert), 0.735)
    if len(y_base) > 1:
        x_base += 0.020 * np.sin(np.arange(len(y_base)) * 1.9)
    if len(y_pert) > 1:
        x_pert += 0.020 * np.cos(np.arange(len(y_pert)) * 1.7)

    axins.scatter(
        x_base,
        y_base,
        s=12,
        c=base_color,
        marker="o",
        edgecolors=(1, 1, 1, 0.90),
        linewidths=0.25,
        alpha=0.92,
        zorder=4,
    )
    axins.scatter(
        x_pert,
        y_pert,
        s=17,
        c=pert_color,
        marker="D",
        edgecolors=(1, 1, 1, 0.92),
        linewidths=0.25,
        alpha=0.96,
        zorder=5,
    )
    axins.plot([0.50, 0.50], [-0.34, 0.33], color="#c7c7c7", lw=0.45, alpha=0.7, zorder=2)
    axins.text(
        0.265,
        0.405,
        f"Control\n{from_fate}",
        ha="center",
        va="center",
        fontsize=4.65,
        weight="bold",
        color=base_color,
        linespacing=0.82,
        zorder=6,
    )
    axins.text(
        0.735,
        0.405,
        f"Perturbed\n{to_fate}",
        ha="center",
        va="center",
        fontsize=4.65,
        weight="bold",
        color=pert_color,
        linespacing=0.82,
        zorder=6,
    )
    axins.set_xlim(0.0, 1.0)
    axins.set_ylim(-0.42, 0.52)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_visible(False)


def make_panel(args: argparse.Namespace) -> dict[str, str]:
    setup_style()
    perturb.seed_all(int(args.seed))
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = runtime_args(args)
    device = perturb.resolve_device(str(args.device))

    summary = perturb.load_json(run_dir / "summary.json")
    adata = ad.read_h5ad(summary["trained_h5ad"])
    adata.obs[rt.annotation_key] = adata.obs[rt.annotation_key].astype("category")
    expression = dense_float32(adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X)
    expr_stats = perturb.expression_stats(expression, float(args.expression_clip_quantile))
    pairs = perturb.observed_time_pairs(adata, rt)
    source_input = perturb.raw_to_input(float(args.source_time), pairs)
    terminal_input = perturb.raw_to_input(float(args.terminal_time), pairs)
    initial_input = float(np.min(np.asarray(adata.obs[rt.time_input_key], dtype=np.float64)))
    spatial_dim = int(adata.obsm[rt.spatial_key].shape[1])
    method = str(summary.get("config", {}).get("ode_method", "rk4"))
    step_size = float(summary.get("config", {}).get("ode_step_size", 0.25))

    model = perturb.load_dynamics(summary, adata, rt, device)
    classifier, label_map, classifier_path = perturb.load_classifier(run_dir, device)
    gat = perturb.load_gat_bundle(adata, device, args.decoder_checkpoint.expanduser().resolve())
    if list(map(str, adata.var_names)) != gat.gene_names:
        raise ValueError("Trained h5ad var_names do not match the GAT checkpoint gene order.")
    knn = perturb.build_terminal_knn(adata, rt, terminal_input, spatial_dim, int(args.knn_k))

    source_indices = perturb.select_source_indices(adata, rt, initial_input, int(args.seed))
    control = perturb.make_control_pack(
        adata=adata,
        args=rt,
        model=model,
        gat=gat,
        classifier=classifier,
        label_map=label_map,
        knn=knn,
        source_input=float(source_input),
        terminal_input=float(terminal_input),
        source_indices=source_indices,
        initial_input=initial_input,
        method=method,
        step_size=step_size,
        spatial_dim=spatial_dim,
        expr_stats=expr_stats,
        device=device,
    )

    gene_to_idx = {str(gene): int(idx) for idx, gene in enumerate(map(str, adata.var_names))}
    if str(args.gene) not in gene_to_idx:
        raise KeyError(f"Gene {args.gene!r} is absent from the current 2000-gene model.")
    pert_gene, perturb_info = perturb.gene_latent_add_shift(
        control=control,
        gene_idx=gene_to_idx[str(args.gene)],
        dose=float(args.dose),
        gat=gat,
        expr_stats=expr_stats,
        spatial_dim=spatial_dim,
        args=rt,
        device=device,
    )
    pert_state0 = control.control_state.detach().clone()
    pert_state0[:, spatial_dim:] = pert_gene

    time_input_grid, base_states = record_rollout(
        model=model,
        state0=control.control_state,
        logw0=control.source_logw,
        source_input=source_input,
        terminal_input=terminal_input,
        method=method,
        step_size=step_size,
        metric_dt=float(args.metric_dt),
        chunk_size=int(args.ode_chunk_size),
        device=device,
    )
    _, pert_states = record_rollout(
        model=model,
        state0=pert_state0,
        logw0=control.source_logw,
        source_input=source_input,
        terminal_input=terminal_input,
        method=method,
        step_size=step_size,
        metric_dt=float(args.metric_dt),
        chunk_size=int(args.ode_chunk_size),
        device=device,
    )

    base_probs, base_labels, classes = perturb.classifier_readout(
        classifier,
        label_map,
        torch.as_tensor(base_states[-1], dtype=torch.float32, device=device),
        terminal_input,
        spatial_dim,
        batch_size=int(args.classifier_batch_size),
    )
    pert_probs, pert_labels, _ = perturb.classifier_readout(
        classifier,
        label_map,
        torch.as_tensor(pert_states[-1], dtype=torch.float32, device=device),
        terminal_input,
        spatial_dim,
        batch_size=int(args.classifier_batch_size),
    )
    embedding_df, tree, ref_xy, ref_mean, ref_std, projection_k, embedding_info = load_state_projector(
        adata,
        args.phate_csv,
        rt.gene_input_key,
        int(args.projection_k),
        embedding=str(args.embedding),
        umap_n_neighbors=int(args.umap_n_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        umap_spread=float(args.umap_spread),
        seed=int(args.seed),
    )
    to_idx = classes.index(str(args.to_fate)) if str(args.to_fate) in classes else int(np.argmax(pert_probs.mean(axis=0)))
    selected, select_mode = choose_cells_with_optional_compaction(
        args=args,
        base_labels=base_labels,
        pert_labels=pert_labels,
        base_probs=base_probs,
        pert_probs=pert_probs,
        to_idx=to_idx,
        base_terminal_states=base_states[-1],
        pert_terminal_states=pert_states[-1],
        spatial_dim=spatial_dim,
        tree=tree,
        ref_xy=ref_xy,
        ref_mean=ref_mean,
        ref_std=ref_std,
        projection_k=projection_k,
    )
    base_xy = project_states_to_phate(
        base_states[:, selected, :],
        spatial_dim=spatial_dim,
        tree=tree,
        ref_xy=ref_xy,
        mean=ref_mean,
        std=ref_std,
        k=projection_k,
    )
    pert_xy = project_states_to_phate(
        pert_states[:, selected, :],
        spatial_dim=spatial_dim,
        tree=tree,
        ref_xy=ref_xy,
        mean=ref_mean,
        std=ref_std,
        k=projection_k,
    )
    plot_base_xy = straightened_paths(base_xy) if str(args.control_path_mode) == "straight" else base_xy.copy()
    plot_pert_xy = pert_xy.copy()
    if bool(args.anchor_perturbed_start):
        plot_pert_xy = (plot_pert_xy + (base_xy[0:1, :, :] - plot_pert_xy[0:1, :, :])).astype(np.float32)

    raw_times = np.interp(time_input_grid, pairs["time_input"].to_numpy(float), pairs["raw_time"].to_numpy(float))
    fig, ax = plt.subplots(figsize=(float(args.fig_width), float(args.fig_height)))
    draw_background(ax, embedding_df, int(args.grid), style=str(args.background_style))
    base_color = CELL_COLORS.get(str(args.from_fate), "#777777")
    pert_color = CELL_COLORS.get(str(args.to_fate), "#111111")
    pert_linestyle = "dashed" if str(args.perturbed_line_style) == "dashed" else "solid"
    base_line_alpha = 0.20 if bool(args.endpoint_emphasis) else 0.30
    pert_line_alpha = 0.32 if bool(args.endpoint_emphasis) else 0.56
    marker_scale = max(float(args.endpoint_marker_scale), 0.2)
    endpoint_display = str(args.endpoint_display)
    for j in range(len(selected)):
        line_with_shadow(ax, plot_base_xy[:, j, :], color=base_color, lw=0.78, alpha=base_line_alpha, zorder=4)
        line_with_shadow(ax, plot_pert_xy[:, j, :], color=pert_color, lw=0.92, alpha=pert_line_alpha, zorder=7, linestyle=pert_linestyle)
        if float(args.endpoint_link_alpha) > 0:
            ax.plot(
                [plot_base_xy[-1, j, 0], plot_pert_xy[-1, j, 0]],
                [plot_base_xy[-1, j, 1], plot_pert_xy[-1, j, 1]],
                color="#2f2f2f",
                lw=0.45,
                alpha=float(args.endpoint_link_alpha),
                zorder=6,
            )
    base_med = np.column_stack([np.median(plot_base_xy[:, :, 0], axis=1), np.median(plot_base_xy[:, :, 1], axis=1)])
    pert_med = np.column_stack([np.median(plot_pert_xy[:, :, 0], axis=1), np.median(plot_pert_xy[:, :, 1], axis=1)])
    ax.plot(base_med[:, 0], base_med[:, 1], color=base_color, lw=2.4, alpha=0.96, solid_capstyle="round", zorder=12)
    ax.plot(
        pert_med[:, 0],
        pert_med[:, 1],
        color=pert_color,
        lw=2.7,
        alpha=0.96,
        linestyle=(0, (3.5, 2.1)) if pert_linestyle == "dashed" else "solid",
        solid_capstyle="round",
        zorder=13,
    )
    arrow_on_curve(ax, base_med, color=base_color, frac=0.66, mutation_scale=9.5, lw=1.2, zorder=20)
    arrow_on_curve(ax, pert_med, color=pert_color, frac=0.66, mutation_scale=10.5, lw=1.3, zorder=21)
    base_end_center: tuple[float, float] | None = None
    pert_end_center: tuple[float, float] | None = None
    if bool(args.endpoint_emphasis):
        if str(args.endpoint_zone_labels) == "direct" or str(args.endpoint_callout) != "none" or endpoint_display == "centroid":
            base_end_center = add_endpoint_zone(
                ax,
                plot_base_xy[-1, :, :],
                color=base_color,
                alpha=float(args.endpoint_zone_alpha),
                zorder=16,
            )
            pert_end_center = add_endpoint_zone(
                ax,
                plot_pert_xy[-1, :, :],
                color=pert_color,
                alpha=float(args.endpoint_zone_alpha),
                zorder=18,
            )
        if endpoint_display == "centroid":
            ax.scatter(plot_base_xy[-1, :, 0], plot_base_xy[-1, :, 1], s=24 * marker_scale, c="white", edgecolors="none", alpha=0.34, zorder=17)
            ax.scatter(plot_pert_xy[-1, :, 0], plot_pert_xy[-1, :, 1], s=30 * marker_scale, c="white", edgecolors="none", alpha=0.38, zorder=19)
        else:
            ax.scatter(plot_base_xy[-1, :, 0], plot_base_xy[-1, :, 1], s=58 * marker_scale, c="white", edgecolors="none", alpha=0.72, zorder=17)
            ax.scatter(plot_pert_xy[-1, :, 0], plot_pert_xy[-1, :, 1], s=76 * marker_scale, c="white", edgecolors="none", alpha=0.78, zorder=19)
    ax.scatter(plot_base_xy[0, :, 0], plot_base_xy[0, :, 1], s=13, c=CELL_COLORS["RGC"], edgecolors="white", linewidths=0.32, alpha=0.80, zorder=15)
    ax.scatter(
        plot_base_xy[-1, :, 0],
        plot_base_xy[-1, :, 1],
        s=(10 * marker_scale) if endpoint_display == "centroid" else ((32 * marker_scale) if bool(args.endpoint_emphasis) else 18),
        c=base_color,
        marker="o",
        edgecolors="white" if endpoint_display == "centroid" else ("#111111" if bool(args.endpoint_emphasis) else "white"),
        linewidths=0.26 if endpoint_display == "centroid" else (0.46 if bool(args.endpoint_emphasis) else 0.30),
        alpha=0.42 if endpoint_display == "centroid" else (0.92 if bool(args.endpoint_emphasis) else 0.68),
        zorder=24,
    )
    ax.scatter(
        plot_pert_xy[-1, :, 0],
        plot_pert_xy[-1, :, 1],
        s=(14 * marker_scale) if endpoint_display == "centroid" else ((44 * marker_scale) if bool(args.endpoint_emphasis) else 22),
        c=pert_color,
        marker="D" if bool(args.endpoint_emphasis) else "o",
        edgecolors="white" if endpoint_display == "centroid" else ("#111111" if bool(args.endpoint_emphasis) else "white"),
        linewidths=0.28 if endpoint_display == "centroid" else (0.50 if bool(args.endpoint_emphasis) else 0.35),
        alpha=0.52 if endpoint_display == "centroid" else (0.96 if bool(args.endpoint_emphasis) else 0.82),
        zorder=26,
    )
    if endpoint_display == "centroid":
        if base_end_center is None:
            base_end_center = tuple(np.nanmedian(plot_base_xy[-1, :, :], axis=0))
        if pert_end_center is None:
            pert_end_center = tuple(np.nanmedian(plot_pert_xy[-1, :, :], axis=0))
        ax.scatter([base_end_center[0]], [base_end_center[1]], s=105 * marker_scale, c="white", marker="o", edgecolors="none", alpha=0.88, zorder=30)
        ax.scatter([pert_end_center[0]], [pert_end_center[1]], s=130 * marker_scale, c="white", marker="D", edgecolors="none", alpha=0.90, zorder=31)
        ax.scatter(
            [base_end_center[0]],
            [base_end_center[1]],
            s=58 * marker_scale,
            c=base_color,
            marker="o",
            edgecolors="#111111",
            linewidths=0.55,
            alpha=0.98,
            zorder=32,
        )
        ax.scatter(
            [pert_end_center[0]],
            [pert_end_center[1]],
            s=78 * marker_scale,
            c=pert_color,
            marker="D",
            edgecolors="#111111",
            linewidths=0.58,
            alpha=0.98,
            zorder=33,
        )
        add_endpoint_centroid_labels(
            ax,
            base_center=base_end_center,
            pert_center=pert_end_center,
            from_fate=str(args.from_fate),
            to_fate=str(args.to_fate),
            base_color=base_color,
            pert_color=pert_color,
        )
    if str(args.endpoint_zone_labels) == "direct":
        if base_end_center is None:
            base_end_center = tuple(np.nanmedian(plot_base_xy[-1, :, :], axis=0))
        if pert_end_center is None:
            pert_end_center = tuple(np.nanmedian(plot_pert_xy[-1, :, :], axis=0))
        add_endpoint_direct_labels(
            ax,
            base_center=base_end_center,
            pert_center=pert_end_center,
            from_fate=str(args.from_fate),
            to_fate=str(args.to_fate),
            base_color=base_color,
            pert_color=pert_color,
        )
    if str(args.endpoint_callout) == "split":
        add_endpoint_split_callout(
            ax,
            plot_base_xy[-1, :, :],
            plot_pert_xy[-1, :, :],
            from_fate=str(args.from_fate),
            to_fate=str(args.to_fate),
            base_color=base_color,
            pert_color=pert_color,
        )
    elif str(args.endpoint_callout) == "strip":
        add_endpoint_strip_callout(
            ax,
            plot_base_xy[-1, :, :],
            plot_pert_xy[-1, :, :],
            from_fate=str(args.from_fate),
            to_fate=str(args.to_fate),
            base_color=base_color,
            pert_color=pert_color,
        )
    if str(args.endpoint_labels) == "badge":
        add_endpoint_badges(
            ax,
            plot_base_xy[-1, :, :],
            plot_pert_xy[-1, :, :],
            from_fate=str(args.from_fate),
            to_fate=str(args.to_fate),
            base_color=base_color,
            pert_color=pert_color,
        )

    label = f"{args.gene} {'+' if float(args.dose) > 0 else ''}{float(args.dose):g}"
    if str(args.title_mode) == "compact":
        ax.text(
            0.018,
            0.982,
            f"{label} perturbation",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            weight="bold",
            color="#111111",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.95)],
        )
        ax.text(
            0.018,
            0.935,
            f"{select_mode}; n={len(selected)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
            color="#555555",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white", alpha=0.95)],
        )
    if str(args.legend_mode) == "none":
        pass
    elif str(args.legend_mode) == "direct":
        start_xy = tuple(np.nanmedian(plot_base_xy[0, :, :], axis=0))
        base_end_xy = tuple(base_med[-1])
        pert_end_xy = tuple(pert_med[-1])
        direct_label(ax, (start_xy[0] + 0.04, start_xy[1] + 0.02), f"E{args.source_time:g} RGC", color=CELL_COLORS["RGC"])
        direct_label(ax, (base_end_xy[0] - 0.08, base_end_xy[1] + 0.04), f"Control: {args.from_fate}", color=base_color, ha="right")
        direct_label(ax, (pert_end_xy[0] - 0.04, pert_end_xy[1] - 0.03), f"{label}: {args.to_fate}", color=pert_color, ha="right")
    else:
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CELL_COLORS["RGC"], markeredgecolor="white", markersize=4.2, label=f"E{args.source_time:g} RGC"),
            Line2D([0], [0], color=base_color, lw=2.2, label=f"control to {args.from_fate}"),
            Line2D([0], [0], color=pert_color, lw=2.4, label=f"{label} to {args.to_fate}"),
        ]
        ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=False, fontsize=5.6, handlelength=1.5)

    condition_tag = f"{str(args.gene)}_{float(args.dose):+g}".replace("+", "plus").replace("-", "minus").replace(".", "p")
    embedding_tag = str(args.embedding)
    if str(args.embedding) == "umap":
        embedding_tag = f"umap_n{int(args.umap_n_neighbors)}_d{float(args.umap_min_dist):g}".replace(".", "p")
    if str(args.background_style) != "potential":
        embedding_tag = f"{embedding_tag}_{args.background_style}"
    base_stem = out_dir / f"paired_trajectory_{condition_tag}_{args.from_fate}_to_{args.to_fate}_{embedding_tag}"
    paths = save_all(fig, base_stem)

    source_rows: list[dict[str, Any]] = []
    for local_j, source_pos in enumerate(selected):
        raw_idx = int(source_indices[source_pos])
        cell_name = str(adata.obs_names[raw_idx])
        for t_idx, (t_input, raw_time) in enumerate(zip(time_input_grid, raw_times)):
            for condition, xy_arr in (("control", plot_base_xy), ("perturbed", plot_pert_xy)):
                source_rows.append(
                    {
                        "cell_id": cell_name,
                        "source_array_index": int(source_pos),
                        "condition": condition,
                        "time_input": float(t_input),
                        "time": float(raw_time),
                        "phate1": float(xy_arr[t_idx, local_j, 0]),
                        "phate2": float(xy_arr[t_idx, local_j, 1]),
                        "baseline_terminal_label": str(base_labels[source_pos]),
                        "perturbed_terminal_label": str(pert_labels[source_pos]),
                        "baseline_P_to_fate": float(base_probs[source_pos, to_idx]),
                        "perturbed_P_to_fate": float(pert_probs[source_pos, to_idx]),
                        "display_control_path_mode": str(args.control_path_mode),
                        "display_perturbed_start": "anchored_to_control" if bool(args.anchor_perturbed_start) else "projected",
                        "display_perturbed_line_style": str(args.perturbed_line_style),
                        "cell_selection": str(args.cell_selection),
                        "endpoint_emphasis": bool(args.endpoint_emphasis),
                        "endpoint_display": str(args.endpoint_display),
                    }
                )
    source_csv = base_stem.with_name(base_stem.name + "_source_data.csv")
    pd.DataFrame(source_rows).to_csv(source_csv, index=False)

    counts = pd.DataFrame(
        {
            "baseline_terminal_label": base_labels,
            "perturbed_terminal_label": pert_labels,
        }
    ).value_counts().reset_index(name="n")
    counts_csv = base_stem.with_name(base_stem.name + "_all_source_transition_counts.csv")
    counts.to_csv(counts_csv, index=False)
    summary_out = {
        "status": "DONE",
        "definition": "Same E12.5 RGC source cells are rolled out under control and under a gene z+x perturbation, then projected to the observed PHATE potential landscape by kNN interpolation from real-cell gene latents.",
        "run_dir": str(run_dir),
        "trained_h5ad": str(summary["trained_h5ad"]),
        "classifier_path": str(classifier_path),
        "gat_checkpoint": str(gat.checkpoint_path),
        "gene": str(args.gene),
        "dose": float(args.dose),
        "source_time": float(args.source_time),
        "terminal_time": float(args.terminal_time),
        "selected_mode": select_mode,
        "n_selected": int(len(selected)),
        "n_source_cells": int(len(source_indices)),
        "from_fate": str(args.from_fate),
        "to_fate": str(args.to_fate),
        "perturbation_info": perturb_info,
        "embedding_info": embedding_info,
        "background_style": str(args.background_style),
        "display_control_path_mode": str(args.control_path_mode),
        "display_perturbed_start": "anchored_to_control" if bool(args.anchor_perturbed_start) else "projected",
        "display_perturbed_line_style": str(args.perturbed_line_style),
        "cell_selection": str(args.cell_selection),
        "endpoint_emphasis": bool(args.endpoint_emphasis),
        "endpoint_labels": str(args.endpoint_labels),
        "endpoint_zone_labels": str(args.endpoint_zone_labels),
        "endpoint_marker_scale": float(args.endpoint_marker_scale),
        "endpoint_callout": str(args.endpoint_callout),
        "endpoint_display": str(args.endpoint_display),
        "outputs": paths,
        "source_data": str(source_csv),
        "transition_counts": str(counts_csv),
    }
    summary_path = base_stem.with_name(base_stem.name + "_summary.json")
    summary_path.write_text(json.dumps(summary_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["source_data"] = str(source_csv)
    paths["summary"] = str(summary_path)
    return paths


def main() -> None:
    args = parse_args()
    paths = make_panel(args)
    print(json.dumps({"status": "DONE", "paths": paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

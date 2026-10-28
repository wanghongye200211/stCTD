#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "SpaPOT_runs"
    / "dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign"
)
DEFAULT_PHATE_CSV = (
    PROJECT_ROOT
    / "result_figures"
    / "dorsal_midbrain_12p5_14p5_16p5_hybrid"
    / "spatial_reconstruction_story_v2"
    / "fig04_observed_expression_potential_embeddings_v1"
    / "observed_expression_pca50_phate_source.csv"
)
DEFAULT_SOURCE_CSV = (
    PROJECT_ROOT
    / "result_figures"
    / "\u5c0f\u9f20\u4e2d\u8111"
    / "g\u6270\u52a8"
    / "paired_split_all1708_source_control_rollout"
    / "paired_trajectory_Efna5_plus6_NeuB_to_GlioB_phate_source_data.csv"
)
DEFAULT_COUNTS_CSV = (
    PROJECT_ROOT
    / "result_figures"
    / "\u5c0f\u9f20\u4e2d\u8111"
    / "g\u6270\u52a8"
    / "paired_split_all1708_source_control_rollout"
    / "paired_trajectory_Efna5_plus6_NeuB_to_GlioB_phate_all_source_transition_counts.csv"
)
DEFAULT_SCREEN_CSV = (
    PROJECT_ROOT
    / "result_figures"
    / "\u5c0f\u9f20\u4e2d\u8111"
    / "g\u6270\u52a8"
    / "final_perturbation_story_v1"
    / "dose_response_screen_all1708"
    / "literature_gene_b_to_b_transition_screen.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "result_figures"
    / "\u5c0f\u9f20\u4e2d\u8111"
    / "g\u6270\u52a8"
    / "final_perturbation_story_v1"
)

CELL_COLORS = {"RGC": "#2ca25f", "NeuB": "#f28e2b", "GlioB": "#4e79a7"}
TEXT = "#151515"
MUTED = "#666666"
GREY = "#9a9a9a"
OE_RED = "#d73027"
DE_BLUE = "#2166ac"
HOUSEKEEPING_PREFIXES = ("Rps", "Rpl", "mt-", "Mt-", "H2a", "H2b", "Hnrnp", "Nduf", "Psm")
HOUSEKEEPING_EXACT = {"Malat1", "Calm1", "Fth1", "Tbca", "Tomm5", "Mrpl28", "Polr3k", "Slc25a5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make final dorsal-midbrain gene perturbation story figures.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--phate-csv", type=Path, default=DEFAULT_PHATE_CSV)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--counts-csv", type=Path, default=DEFAULT_COUNTS_CSV)
    parser.add_argument("--screen-csv", type=Path, default=DEFAULT_SCREEN_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--layer", default="lognorm")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--expression-bins", type=int, default=72)
    parser.add_argument("--trajectory-x-compression", type=float, default=2.15)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Arial Unicode MS", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, stem: Path, dpi: int, *, tight: bool = True) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg", "tiff"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if tight:
            kwargs.update({"bbox_inches": "tight", "pad_inches": 0.025})
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        outputs[suffix] = str(path)
    plt.close(fig)
    return outputs


def dense_float32(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def read_expression(adata: ad.AnnData, layer: str) -> np.ndarray:
    if layer in adata.layers:
        return dense_float32(adata.layers[layer])
    if layer == "X":
        return dense_float32(adata.X)
    raise KeyError(f"Missing expression layer: {layer}")


def condition_tensor(df: pd.DataFrame, condition: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    sub = df.loc[df["condition"].astype(str) == condition].copy()
    sub = sub.sort_values(["cell_id", "time_input"])
    cells = list(dict.fromkeys(sub["cell_id"].astype(str)))
    times = np.sort(sub["time_input"].unique())
    xy = np.full((len(times), len(cells), 2), np.nan, dtype=np.float32)
    cell_to_i = {cell: i for i, cell in enumerate(cells)}
    time_to_i = {float(t): i for i, t in enumerate(times)}
    for row in sub.itertuples(index=False):
        xy[time_to_i[float(row.time_input)], cell_to_i[str(row.cell_id)], 0] = float(row.phate1)
        xy[time_to_i[float(row.time_input)], cell_to_i[str(row.cell_id)], 1] = float(row.phate2)
    return xy, cells, times


def local_bounds(xy: np.ndarray, margin_x: float = 0.24, margin_y: float = 0.18) -> tuple[float, float, float, float]:
    flat = np.asarray(xy, dtype=float).reshape(-1, 2)
    flat = flat[np.isfinite(flat).all(axis=1)]
    xlo, xhi = float(np.min(flat[:, 0])), float(np.max(flat[:, 0]))
    ylo, yhi = float(np.min(flat[:, 1])), float(np.max(flat[:, 1]))
    xr = max(xhi - xlo, 1e-8)
    yr = max(yhi - ylo, 1e-8)
    return xlo - margin_x * xr, xhi + margin_x * xr, ylo - margin_y * yr, yhi + margin_y * yr


def draw_local_background(ax: plt.Axes, phate: pd.DataFrame, view_xy: np.ndarray, x_compression: float) -> None:
    x = phate["pca50_phate1"].to_numpy(float)
    y = phate["pca50_phate2"].to_numpy(float)
    xlo, xhi, ylo, yhi = local_bounds(view_xy)
    coords = np.column_stack([x, y])
    kth = cKDTree(coords).query(coords, k=16)[0][:, -1]
    dense = kth <= np.quantile(kth, 0.93)
    mask = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi) & dense
    ax.scatter(x[mask], y[mask], s=4.0, c=GREY, alpha=0.18, linewidths=0, rasterized=True, zorder=0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect(float(x_compression), adjustable="box")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_rollout_lines(ax: plt.Axes, xy: np.ndarray, color: str, linestyle: str) -> None:
    ls = (0, (3.0, 2.0)) if linestyle == "dashed" else "solid"
    for j in range(xy.shape[1]):
        ax.plot(
            xy[:, j, 0],
            xy[:, j, 1],
            color=color,
            lw=0.84,
            alpha=0.34,
            linestyle=ls,
            solid_capstyle="round",
            zorder=8,
        )


def make_trajectory_pair(args: argparse.Namespace, phate: pd.DataFrame) -> dict[str, str]:
    source = pd.read_csv(args.source_csv.expanduser().resolve())
    control_xy, control_cells, _ = condition_tensor(source, "control")
    pert_xy, pert_cells, _ = condition_tensor(source, "perturbed")
    if control_cells != pert_cells:
        raise ValueError("Control and perturbed cells are not identical.")
    view_xy = np.concatenate([control_xy.reshape(-1, 2), pert_xy.reshape(-1, 2)], axis=0)
    n_cells = len(control_cells)

    fig, axes = plt.subplots(2, 1, figsize=(4.15, 6.0), sharex=False, sharey=False)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.035, hspace=0.08)
    panels = [
        (axes[0], control_xy, CELL_COLORS["NeuB"], "solid", "Control rollout", "same E12.5 RGC cells, baseline NeuB-bound"),
        (
            axes[1],
            pert_xy,
            CELL_COLORS["GlioB"],
            "dashed",
            "Efna5 +6 overexpression-like rollout",
            "same cells after gene z+x perturbation, shifted to GlioB",
        ),
    ]
    for ax, xy, color, linestyle, title, subtitle in panels:
        draw_local_background(ax, phate, view_xy, float(args.trajectory_x_compression))
        draw_rollout_lines(ax, xy, color, linestyle)
        ax.scatter(
            xy[0, :, 0],
            xy[0, :, 1],
            s=21,
            c=CELL_COLORS["RGC"],
            edgecolors="white",
            linewidths=0.45,
            alpha=0.94,
            zorder=20,
        )
        ax.scatter(
            xy[-1, :, 0],
            xy[-1, :, 1],
            s=24,
            c=color,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.92,
            zorder=21,
        )
        ax.text(
            0.015,
            0.965,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            weight="bold",
            color=TEXT,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white", alpha=0.95)],
        )
        ax.text(
            0.015,
            0.895,
            f"{subtitle}; n={n_cells}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.9,
            color=MUTED,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.95)],
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CELL_COLORS["RGC"], markeredgecolor="white", markersize=5.4, label="RGC start"),
        Line2D([0], [0], color=CELL_COLORS["NeuB"], lw=1.6, label="control to NeuB"),
        Line2D([0], [0], color=CELL_COLORS["GlioB"], lw=1.6, linestyle=(0, (3, 2)), label="perturbed to GlioB"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.52, 0.0005), fontsize=6.1, handlelength=1.8)
    return save_all(fig, args.out_dir / "02_Efna5_control_vs_perturbed_rollouts", int(args.dpi), tight=False)


def make_screen_heatmap(args: argparse.Namespace) -> dict[str, str]:
    screen = pd.read_csv(args.screen_csv.expanduser().resolve())
    screen = screen.copy()
    screen["signed_switch_pct"] = 100.0 * (screen["NeuB_to_GlioB"] - screen["GlioB_to_NeuB"]) / screen["n_source_cells"]
    screen["max_switch_pct"] = 100.0 * screen[["NeuB_to_GlioB", "GlioB_to_NeuB"]].max(axis=1) / screen["n_source_cells"]
    genes = (
        screen.groupby("gene")["max_switch_pct"]
        .max()
        .sort_values(ascending=False)
        .head(12)
        .index.astype(str)
        .tolist()
    )
    doses = sorted(screen["dose"].dropna().unique())
    matrix = np.zeros((len(genes), len(doses)), dtype=float)
    annot = np.zeros_like(matrix)
    for i, gene in enumerate(genes):
        for j, dose in enumerate(doses):
            sub = screen[(screen["gene"].astype(str) == gene) & (np.isclose(screen["dose"], dose))]
            if sub.empty:
                matrix[i, j] = np.nan
                annot[i, j] = np.nan
            else:
                matrix[i, j] = float(sub["signed_switch_pct"].iloc[0])
                annot[i, j] = float(sub["max_switch_pct"].iloc[0])

    vmax = max(2.0, float(np.nanmax(np.abs(matrix))))
    cmap = LinearSegmentedColormap.from_list("orange_white_blue", [CELL_COLORS["NeuB"], "#f7f7f7", CELL_COLORS["GlioB"]])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    fig.subplots_adjust(left=0.17, right=0.91, top=0.86, bottom=0.18)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontstyle="italic")
    ax.set_xticks(np.arange(len(doses)))
    ax.set_xticklabels([f"{d:g}" for d in doses])
    ax.set_xlabel("signed perturbation dose")
    ax.set_title("Gene perturbation screen", pad=5, fontsize=8.5, weight="bold")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(annot[i, j]) and annot[i, j] >= 1.0:
                color = "white" if abs(matrix[i, j]) > 0.56 * vmax else "#1b1b1b"
                ax.text(j, i, f"{annot[i, j]:.0f}", ha="center", va="center", fontsize=5.3, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("switch direction (%)", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    ax.text(
        0.0,
        -0.23,
        "blue: NeuB->GlioB; orange: GlioB->NeuB; numbers: max switch %",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=MUTED,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return save_all(fig, args.out_dir / "01_gene_perturbation_screen_heatmap", int(args.dpi))


def robust_scale_rows(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        row = values[i].astype(float)
        lo, hi = np.nanquantile(row, [0.03, 0.97])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(row)), float(np.nanmax(row))
        if hi <= lo:
            out[i] = 0.0
        else:
            out[i] = np.clip((row - lo) / (hi - lo), 0.0, 1.0)
    return out


def make_expression_heatmap(args: argparse.Namespace, phate: pd.DataFrame) -> dict[str, str]:
    summary = json.loads((args.run_dir.expanduser().resolve() / "summary.json").read_text(encoding="utf-8"))
    adata = ad.read_h5ad(summary["trained_h5ad"])
    expr = read_expression(adata, str(args.layer))
    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names.astype(str))}
    obs_to_idx = {str(name): i for i, name in enumerate(adata.obs_names.astype(str))}

    gene_order = [
        "Efna5",
        "Dbi",
        "Adgrb3",
        "Ptn",
        "Fzd3",
        "Dbx1",
        "Fabp7",
        "Vim",
        "Nedd4",
        "Mdk",
        "St18",
        "Sox6",
        "Cdkn1c",
        "Nfat5",
        "Glis3",
        "Zic4",
        "Dut",
        "Banf1",
        "Slc25a5",
        "Zfhx4",
    ]
    genes = [gene for gene in gene_order if gene in var_to_idx]
    if len(genes) < 3:
        raise ValueError("Too few requested genes are present in the model gene set.")

    phate = phate.copy()
    if "label" not in phate.columns and "cell_type" in phate.columns:
        phate["label"] = phate["cell_type"].astype(str)
    phate["adata_idx"] = phate["adata_obs_name"].astype(str).map(obs_to_idx)
    phate = phate.dropna(subset=["adata_idx", "potential_U"]).copy()
    phate["adata_idx"] = phate["adata_idx"].astype(int)
    phate = phate.sort_values("potential_U", ascending=False).reset_index(drop=True)
    bins = np.array_split(np.arange(phate.shape[0]), int(args.expression_bins))
    raw = np.zeros((len(genes), len(bins)), dtype=float)
    label_frac = np.zeros((3, len(bins)), dtype=float)
    labels = ["RGC", "NeuB", "GlioB"]
    label_values = phate["label"].astype(str).to_numpy()
    for b, idx in enumerate(bins):
        ad_idx = phate.loc[idx, "adata_idx"].to_numpy(int)
        for gi, gene in enumerate(genes):
            raw[gi, b] = float(np.mean(expr[ad_idx, var_to_idx[gene]]))
        for li, label in enumerate(labels):
            label_frac[li, b] = float(np.mean(label_values[idx] == label))
    raw = gaussian_filter1d(raw, sigma=1.15, axis=1, mode="nearest")
    mat = robust_scale_rows(raw)

    fig = plt.figure(figsize=(3.65, 5.05))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.24, 4.9], hspace=0.03, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.23, right=0.87, top=0.84, bottom=0.09)
    fig.text(0.23, 0.965, "Candidate genes along potential", ha="left", va="top", fontsize=8.8, weight="bold", color=TEXT)

    x = np.arange(label_frac.shape[1])
    bottom = np.zeros_like(x, dtype=float)
    for label in labels:
        frac = label_frac[labels.index(label)]
        ax_top.bar(x, frac, bottom=bottom, width=1.0, color=CELL_COLORS[label], linewidth=0, alpha=0.95)
        bottom += frac
    ax_top.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontstyle="italic")
    ax.set_xticks([0, mat.shape[1] - 1])
    ax.set_xticklabels(["high potential", "low potential"])
    ax.set_xlabel("observed cells ranked by potential")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.set_label("scaled expression", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    ax_top.annotate(
        "",
        xy=(0.02, 1.85),
        xytext=(0.98, 1.85),
        xycoords=ax_top.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"},
        annotation_clip=False,
    )
    ax_top.text(
        0.5,
        2.1,
        "potential-ordered observed program",
        transform=ax_top.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#333333",
    )
    return save_all(fig, args.out_dir / "03_candidate_gene_expression_by_potential", int(args.dpi))


def candidate_gene_order() -> list[str]:
    return [
        "Efna5",
        "Dbi",
        "Adgrb3",
        "Ptn",
        "Fzd3",
        "Dbx1",
        "Fabp7",
        "Vim",
        "Nedd4",
        "Mdk",
        "St18",
        "Sox6",
        "Cdkn1c",
        "Nfat5",
        "Glis3",
        "Zic4",
        "Dut",
        "Banf1",
        "Slc25a5",
        "Zfhx4",
    ]


def is_housekeeping_like(gene: str) -> bool:
    return str(gene) in HOUSEKEEPING_EXACT or str(gene).startswith(HOUSEKEEPING_PREFIXES)


def load_expression_context(args: argparse.Namespace) -> tuple[ad.AnnData, np.ndarray, dict[str, int], dict[str, int]]:
    summary = json.loads((args.run_dir.expanduser().resolve() / "summary.json").read_text(encoding="utf-8"))
    adata = ad.read_h5ad(summary["trained_h5ad"])
    expr = read_expression(adata, str(args.layer))
    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names.astype(str))}
    obs_to_idx = {str(name): i for i, name in enumerate(adata.obs_names.astype(str))}
    return adata, expr, var_to_idx, obs_to_idx


def two_fate_gene_stats(adata: ad.AnnData, expr: np.ndarray) -> pd.DataFrame:
    labels = adata.obs["Annotation"].astype(str).to_numpy()
    neu_mask = labels == "NeuB"
    glio_mask = labels == "GlioB"
    rows = []
    for j, gene in enumerate(adata.var_names.astype(str)):
        neu = float(np.mean(expr[neu_mask, j]))
        glio = float(np.mean(expr[glio_mask, j]))
        neu_frac = float(np.mean(expr[neu_mask, j] > 0))
        glio_frac = float(np.mean(expr[glio_mask, j] > 0))
        marker_score = abs(glio - neu) * max(neu_frac, glio_frac)
        log2fc = float(np.log2((glio + 1e-4) / (neu + 1e-4)))
        rows.append(
            {
                "gene": str(gene),
                "NeuB_mean": neu,
                "GlioB_mean": glio,
                "NeuB_frac": neu_frac,
                "GlioB_frac": glio_frac,
                "max_nonzero_frac": max(neu_frac, glio_frac),
                "marker_score": marker_score,
                "log2FC_GlioB_vs_NeuB": log2fc,
                "is_housekeeping_like": is_housekeeping_like(str(gene)),
                "is_perturbation_candidate": str(gene) in set(candidate_gene_order()),
            }
        )
    stats = pd.DataFrame(rows)
    stats["direction_score"] = stats["log2FC_GlioB_vs_NeuB"].abs() * np.sqrt(stats["marker_score"].clip(lower=0.0))
    return stats


def selected_two_fate_genes(stats: pd.DataFrame, n_each: int = 12) -> list[str]:
    usable = stats[
        (~stats["is_housekeeping_like"].astype(bool))
        & (stats["max_nonzero_frac"] >= 0.08)
        & (stats["marker_score"] >= 0.006)
    ].copy()
    glio_seed = ["Glis3", "Rmst", "Wls", "Efna5", "Zfhx4", "Zic4", "Vim", "Dbi", "Mdk", "Nedd4"]
    neu_seed = ["Lmo1", "Tal2", "Dach1", "Mpped2", "Setbp1", "Fabp7", "Cdkn1c", "Sox11", "Adgrb3", "Ptn"]

    def complete(seed: list[str], frame: pd.DataFrame, ascending: bool) -> list[str]:
        selected = [g for g in seed if g in set(frame["gene"].astype(str))]
        ranked = frame.sort_values("direction_score", ascending=False)["gene"].astype(str).tolist()
        for gene in ranked:
            if gene not in selected:
                selected.append(gene)
            if len(selected) >= int(n_each):
                break
        return selected[: int(n_each)]

    glio = complete(glio_seed, usable[usable["log2FC_GlioB_vs_NeuB"] > 0.12], ascending=False)
    neu = complete(neu_seed, usable[usable["log2FC_GlioB_vs_NeuB"] < -0.12], ascending=True)
    return glio + neu


def make_two_fate_stories_heatmap(args: argparse.Namespace, phate: pd.DataFrame) -> dict[str, str]:
    adata, expr, var_to_idx, obs_to_idx = load_expression_context(args)
    stats = two_fate_gene_stats(adata, expr)
    genes = [gene for gene in selected_two_fate_genes(stats) if gene in var_to_idx]
    stat_by_gene = stats.set_index("gene")

    cur = phate.copy()
    if "label" not in cur.columns and "cell_type" in cur.columns:
        cur["label"] = cur["cell_type"].astype(str)
    cur["adata_idx"] = cur["adata_obs_name"].astype(str).map(obs_to_idx)
    cur = cur[cur["label"].astype(str).isin(["NeuB", "GlioB"])].dropna(subset=["adata_idx", "potential_U"]).copy()
    cur["adata_idx"] = cur["adata_idx"].astype(int)
    neu = cur[cur["label"].astype(str) == "NeuB"].sort_values("potential_U", ascending=False).reset_index(drop=True)
    glio = cur[cur["label"].astype(str) == "GlioB"].sort_values("potential_U", ascending=False).reset_index(drop=True)
    bins_each = 42
    bins = [("NeuB", idx) for idx in np.array_split(np.arange(neu.shape[0]), bins_each)]
    bins += [("GlioB", idx) for idx in np.array_split(np.arange(glio.shape[0]), bins_each)]
    raw = np.zeros((len(genes), len(bins)), dtype=float)
    for b, (label, idx) in enumerate(bins):
        frame = neu if label == "NeuB" else glio
        ad_idx = frame.loc[idx, "adata_idx"].to_numpy(int)
        for gi, gene in enumerate(genes):
            raw[gi, b] = float(np.mean(expr[ad_idx, var_to_idx[gene]]))
    mat = robust_scale_rows(gaussian_filter1d(raw, sigma=1.05, axis=1, mode="nearest"))
    divider = bins_each - 0.5

    fig = plt.figure(figsize=(3.45, 5.15))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.20, 4.9], hspace=0.035, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.25, right=0.86, top=0.88, bottom=0.095)
    fig.text(0.25, 0.972, "Two-fate expression program", ha="left", va="top", fontsize=8.8, weight="bold", color=TEXT)

    top_colors = [CELL_COLORS["NeuB"]] * bins_each + [CELL_COLORS["GlioB"]] * bins_each
    for x, color in enumerate(top_colors):
        ax_top.add_patch(Rectangle((x - 0.5, 0), 1.0, 1.0, facecolor=color, edgecolor=color, linewidth=0))
    ax_top.set_xlim(-0.5, len(bins) - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axvline(divider, color="white", lw=1.2)
    ax_top.axis("off")

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
    ax.axvline(divider, color="white", lw=1.3)
    ax.axhline(11.5, color="white", lw=1.0)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontstyle="italic")
    ax.set_xticks([bins_each / 2 - 0.5, bins_each + bins_each / 2 - 0.5])
    ax.set_xticklabels(["NeuB", "GlioB"])
    ax.set_xlabel("observed B cells grouped by terminal type")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.set_label("scaled expression", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)

    ax.text(0.995, 0.77, "GlioB-enriched", transform=ax.transAxes, ha="right", va="center", fontsize=5.6, color=CELL_COLORS["GlioB"], weight="bold")
    ax.text(0.005, 0.25, "NeuB-enriched", transform=ax.transAxes, ha="left", va="center", fontsize=5.6, color=CELL_COLORS["NeuB"], weight="bold")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for gene in genes:
        source_rows.append(
            {
                "gene": gene,
                "row_group": "GlioB_enriched" if float(stat_by_gene.loc[gene, "log2FC_GlioB_vs_NeuB"]) > 0 else "NeuB_enriched",
                "log2FC_GlioB_vs_NeuB": float(stat_by_gene.loc[gene, "log2FC_GlioB_vs_NeuB"]),
                "marker_score": float(stat_by_gene.loc[gene, "marker_score"]),
                "is_perturbation_candidate": bool(stat_by_gene.loc[gene, "is_perturbation_candidate"]),
            }
        )
    pd.DataFrame(source_rows).to_csv(out_dir / "03d_two_fate_stories_heatmap_gene_source_data.csv", index=False)
    pd.DataFrame(mat, index=genes).to_csv(out_dir / "03d_two_fate_stories_heatmap_matrix_scaled.csv")
    return save_all(fig, out_dir / "03d_two_fate_STORIES_style_expression_heatmap", int(args.dpi))


def two_class_gene_stats(adata: ad.AnnData, expr: np.ndarray, left_label: str, right_label: str) -> pd.DataFrame:
    labels = adata.obs["Annotation"].astype(str).to_numpy()
    left_mask = labels == str(left_label)
    right_mask = labels == str(right_label)
    rows = []
    for j, gene in enumerate(adata.var_names.astype(str)):
        left = float(np.mean(expr[left_mask, j]))
        right = float(np.mean(expr[right_mask, j]))
        left_frac = float(np.mean(expr[left_mask, j] > 0))
        right_frac = float(np.mean(expr[right_mask, j] > 0))
        marker_score = abs(right - left) * max(left_frac, right_frac)
        log2fc = float(np.log2((right + 1e-4) / (left + 1e-4)))
        rows.append(
            {
                "gene": str(gene),
                "left_label": str(left_label),
                "right_label": str(right_label),
                "left_mean": left,
                "right_mean": right,
                "left_frac": left_frac,
                "right_frac": right_frac,
                "max_nonzero_frac": max(left_frac, right_frac),
                "marker_score": marker_score,
                "log2FC_right_vs_left": log2fc,
                "is_housekeeping_like": is_housekeeping_like(str(gene)),
                "is_perturbation_candidate": str(gene) in set(candidate_gene_order()),
            }
        )
    stats = pd.DataFrame(rows)
    stats["direction_score"] = stats["log2FC_right_vs_left"].abs() * np.sqrt(stats["marker_score"].clip(lower=0.0))
    return stats


def selected_branch_genes(stats: pd.DataFrame, right_label: str, n_each: int = 12) -> list[str]:
    usable = stats[
        (~stats["is_housekeeping_like"].astype(bool))
        & (stats["max_nonzero_frac"] >= 0.08)
        & (stats["marker_score"] >= 0.006)
    ].copy()
    seed_right = {
        "GlioB": ["Glis3", "Rmst", "Wls", "Efna5", "Zfhx4", "Zic4", "Vim", "Dbi", "Mdk", "Nedd4"],
        "NeuB": ["Lmo1", "Tal2", "Dach1", "Mpped2", "Setbp1", "Fabp7", "Cdkn1c", "Sox11", "Adgrb3", "Ptn"],
    }.get(str(right_label), [])
    seed_left = {
        "GlioB": ["Cpe", "Hopx", "Notch2", "Sox2", "Fabp7", "Ptprz1", "Ptn", "Id3", "Vim", "Dbi"],
        "NeuB": ["Cpe", "Hopx", "Notch2", "Sox2", "Ptprz1", "Id3", "Vim", "Dbi", "Efna5", "Zfhx4"],
    }.get(str(right_label), [])

    def complete(seed: list[str], frame: pd.DataFrame) -> list[str]:
        selected = [g for g in seed if g in set(frame["gene"].astype(str))]
        ranked = frame.sort_values("direction_score", ascending=False)["gene"].astype(str).tolist()
        for gene in ranked:
            if gene not in selected:
                selected.append(gene)
            if len(selected) >= int(n_each):
                break
        return selected[: int(n_each)]

    right = complete(seed_right, usable[usable["log2FC_right_vs_left"] > 0.12])
    left = complete(seed_left, usable[usable["log2FC_right_vs_left"] < -0.12])
    return right + left


def make_rgc_branch_stories_heatmap(args: argparse.Namespace, phate: pd.DataFrame, right_label: str) -> dict[str, str]:
    adata, expr, var_to_idx, obs_to_idx = load_expression_context(args)
    left_label = "RGC"
    stats = two_class_gene_stats(adata, expr, left_label, str(right_label))
    genes = [gene for gene in selected_branch_genes(stats, str(right_label)) if gene in var_to_idx]
    stat_by_gene = stats.set_index("gene")

    cur = phate.copy()
    if "label" not in cur.columns and "cell_type" in cur.columns:
        cur["label"] = cur["cell_type"].astype(str)
    cur["adata_idx"] = cur["adata_obs_name"].astype(str).map(obs_to_idx)
    cur = cur[cur["label"].astype(str).isin([left_label, str(right_label)])].dropna(subset=["adata_idx", "potential_U"]).copy()
    cur["adata_idx"] = cur["adata_idx"].astype(int)
    left = cur[cur["label"].astype(str) == left_label].sort_values("potential_U", ascending=False).reset_index(drop=True)
    right = cur[cur["label"].astype(str) == str(right_label)].sort_values("potential_U", ascending=False).reset_index(drop=True)
    bins_each = 42
    bins = [(left_label, idx) for idx in np.array_split(np.arange(left.shape[0]), bins_each)]
    bins += [(str(right_label), idx) for idx in np.array_split(np.arange(right.shape[0]), bins_each)]
    raw = np.zeros((len(genes), len(bins)), dtype=float)
    for b, (label, idx) in enumerate(bins):
        frame = left if label == left_label else right
        ad_idx = frame.loc[idx, "adata_idx"].to_numpy(int)
        for gi, gene in enumerate(genes):
            raw[gi, b] = float(np.mean(expr[ad_idx, var_to_idx[gene]]))
    mat = robust_scale_rows(gaussian_filter1d(raw, sigma=1.05, axis=1, mode="nearest"))
    divider = bins_each - 0.5
    target_color = CELL_COLORS[str(right_label)]

    fig = plt.figure(figsize=(3.45, 5.15))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.20, 4.9], hspace=0.035, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.25, right=0.86, top=0.88, bottom=0.095)
    fig.text(
        0.25,
        0.972,
        f"RGC/{right_label} expression program",
        ha="left",
        va="top",
        fontsize=8.8,
        weight="bold",
        color=TEXT,
    )

    top_colors = [CELL_COLORS[left_label]] * bins_each + [target_color] * bins_each
    for x, color in enumerate(top_colors):
        ax_top.add_patch(Rectangle((x - 0.5, 0), 1.0, 1.0, facecolor=color, edgecolor=color, linewidth=0))
    ax_top.set_xlim(-0.5, len(bins) - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axvline(divider, color="white", lw=1.2)
    ax_top.axis("off")

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
    ax.axvline(divider, color="white", lw=1.3)
    ax.axhline(11.5, color="white", lw=1.0)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontstyle="italic")
    ax.set_xticks([bins_each / 2 - 0.5, bins_each + bins_each / 2 - 0.5])
    ax.set_xticklabels([left_label, str(right_label)])
    ax.set_xlabel(f"observed {left_label}/{right_label} cells grouped by type")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.set_label("scaled expression", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    ax.text(
        0.995,
        0.77,
        f"{right_label}-enriched",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=5.6,
        color=target_color,
        weight="bold",
    )
    ax.text(
        0.005,
        0.25,
        "RGC-enriched",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.6,
        color=CELL_COLORS[left_label],
        weight="bold",
    )

    safe = str(right_label).replace("/", "_")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for gene in genes:
        source_rows.append(
            {
                "gene": gene,
                "row_group": f"{right_label}_enriched" if float(stat_by_gene.loc[gene, "log2FC_right_vs_left"]) > 0 else "RGC_enriched",
                "log2FC_right_vs_RGC": float(stat_by_gene.loc[gene, "log2FC_right_vs_left"]),
                "marker_score": float(stat_by_gene.loc[gene, "marker_score"]),
                "is_perturbation_candidate": bool(stat_by_gene.loc[gene, "is_perturbation_candidate"]),
            }
        )
    pd.DataFrame(source_rows).to_csv(out_dir / f"03e_RGC_{safe}_branch_heatmap_gene_source_data.csv", index=False)
    pd.DataFrame(mat, index=genes).to_csv(out_dir / f"03e_RGC_{safe}_branch_heatmap_matrix_scaled.csv")
    return save_all(fig, out_dir / f"03e_RGC_{safe}_STORIES_style_expression_heatmap", int(args.dpi))


def make_two_type_expression_heatmap(args: argparse.Namespace, phate: pd.DataFrame) -> dict[str, str]:
    _adata, expr, var_to_idx, obs_to_idx = load_expression_context(args)
    genes = [gene for gene in candidate_gene_order() if gene in var_to_idx]

    cur = phate.copy()
    if "label" not in cur.columns and "cell_type" in cur.columns:
        cur["label"] = cur["cell_type"].astype(str)
    cur["adata_idx"] = cur["adata_obs_name"].astype(str).map(obs_to_idx)
    cur = cur[cur["label"].astype(str).isin(["NeuB", "GlioB"])].dropna(subset=["adata_idx", "potential_U"]).copy()
    cur["adata_idx"] = cur["adata_idx"].astype(int)
    cur = cur.sort_values("potential_U", ascending=False).reset_index(drop=True)

    bins = np.array_split(np.arange(cur.shape[0]), max(16, min(int(args.expression_bins), cur.shape[0] // 4)))
    raw = np.zeros((len(genes), len(bins)), dtype=float)
    label_frac = np.zeros((2, len(bins)), dtype=float)
    label_values = cur["label"].astype(str).to_numpy()
    for b, idx in enumerate(bins):
        ad_idx = cur.loc[idx, "adata_idx"].to_numpy(int)
        for gi, gene in enumerate(genes):
            raw[gi, b] = float(np.mean(expr[ad_idx, var_to_idx[gene]]))
        label_frac[0, b] = float(np.mean(label_values[idx] == "NeuB"))
        label_frac[1, b] = float(np.mean(label_values[idx] == "GlioB"))

    neu_mask = cur["label"].astype(str).to_numpy() == "NeuB"
    glio_mask = cur["label"].astype(str).to_numpy() == "GlioB"
    neu_idx = cur.loc[neu_mask, "adata_idx"].to_numpy(int)
    glio_idx = cur.loc[glio_mask, "adata_idx"].to_numpy(int)
    effect = []
    for gene in genes:
        j = var_to_idx[gene]
        neu = float(np.mean(expr[neu_idx, j]))
        glio = float(np.mean(expr[glio_idx, j]))
        effect.append(np.log2((glio + 1e-4) / (neu + 1e-4)))
    order = np.argsort(effect)[::-1]
    genes = [genes[i] for i in order]
    raw = raw[order, :]
    mat = robust_scale_rows(gaussian_filter1d(raw, sigma=1.05, axis=1, mode="nearest"))

    fig = plt.figure(figsize=(3.6, 4.75))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.22, 4.7], hspace=0.03, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.23, right=0.87, top=0.84, bottom=0.09)
    fig.text(0.23, 0.965, "NeuB/GlioB genes along potential", ha="left", va="top", fontsize=8.8, weight="bold", color=TEXT)

    x = np.arange(label_frac.shape[1])
    ax_top.bar(x, label_frac[0], width=1.0, color=CELL_COLORS["NeuB"], linewidth=0, alpha=0.95)
    ax_top.bar(x, label_frac[1], bottom=label_frac[0], width=1.0, color=CELL_COLORS["GlioB"], linewidth=0, alpha=0.95)
    ax_top.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontstyle="italic")
    ax.set_xticks([0, mat.shape[1] - 1])
    ax.set_xticklabels(["high potential", "low potential"])
    ax.set_xlabel("NeuB/GlioB cells ranked by potential")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.set_label("scaled expression", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    ax_top.annotate(
        "",
        xy=(0.02, 1.85),
        xytext=(0.98, 1.85),
        xycoords=ax_top.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"},
        annotation_clip=False,
    )
    ax_top.text(
        0.5,
        2.1,
        "B-cell potential-ordered program",
        transform=ax_top.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#333333",
    )
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CELL_COLORS["NeuB"], markeredgecolor="none", markersize=5.0, label="NeuB"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CELL_COLORS["GlioB"], markeredgecolor="none", markersize=5.0, label="GlioB"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=2, fontsize=5.9)
    return save_all(fig, args.out_dir / "03b_NeuB_GlioB_expression_by_potential", int(args.dpi))


def make_two_type_expression_contrast(args: argparse.Namespace) -> dict[str, str]:
    adata, expr, var_to_idx, _obs_to_idx = load_expression_context(args)
    labels = adata.obs["Annotation"].astype(str).to_numpy()
    genes = [gene for gene in candidate_gene_order() if gene in var_to_idx]
    rows = []
    neu_mask = labels == "NeuB"
    glio_mask = labels == "GlioB"
    for gene in genes:
        j = var_to_idx[gene]
        neu = float(np.mean(expr[neu_mask, j]))
        glio = float(np.mean(expr[glio_mask, j]))
        neu_frac = float(np.mean(expr[neu_mask, j] > 0))
        glio_frac = float(np.mean(expr[glio_mask, j] > 0))
        rows.append(
            {
                "gene": gene,
                "NeuB_mean": neu,
                "GlioB_mean": glio,
                "log2FC_GlioB_vs_NeuB": float(np.log2((glio + 1e-4) / (neu + 1e-4))),
                "max_nonzero_frac": max(neu_frac, glio_frac),
            }
        )
    stats = pd.DataFrame(rows).sort_values("log2FC_GlioB_vs_NeuB", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.65, 4.25))
    fig.subplots_adjust(left=0.25, right=0.96, top=0.88, bottom=0.12)
    y = np.arange(stats.shape[0])
    values = stats["log2FC_GlioB_vs_NeuB"].to_numpy(float)
    colors = np.where(values >= 0, CELL_COLORS["GlioB"], CELL_COLORS["NeuB"])
    ax.axvline(0, color="#555555", lw=0.8)
    for yi, val, color in zip(y, values, colors):
        ax.plot([0, val], [yi, yi], color=color, lw=1.2, alpha=0.8)
    ax.scatter(
        values,
        y,
        s=18 + 70 * stats["max_nonzero_frac"].to_numpy(float),
        c=colors,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(stats["gene"], fontstyle="italic")
    ax.set_xlabel("log2 mean expression fold-change\nGlioB vs NeuB")
    ax.set_title("NeuB/GlioB candidate gene contrast", fontsize=8.6, weight="bold", pad=5)
    ax.grid(axis="x", color="#dddddd", lw=0.45, alpha=0.7)
    ax.tick_params(axis="both", labelsize=6.2, length=2.5)
    lim = max(0.5, float(np.nanmax(np.abs(values))) * 1.14)
    ax.set_xlim(-lim, lim)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CELL_COLORS["NeuB"], markeredgecolor="white", markersize=5.2, label="NeuB-enriched"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CELL_COLORS["GlioB"], markeredgecolor="white", markersize=5.2, label="GlioB-enriched"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=5.8)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "03c_NeuB_GlioB_candidate_gene_contrast_source_data.csv", index=False)
    return save_all(fig, out_dir / "03c_NeuB_GlioB_candidate_gene_contrast", int(args.dpi))


def make_dose_response(args: argparse.Namespace) -> dict[str, str]:
    screen = pd.read_csv(args.screen_csv.expanduser().resolve()).copy()
    gene = "Efna5"
    sub = screen[screen["gene"].astype(str) == gene].copy()
    if sub.empty:
        raise ValueError(f"{gene} was not found in {args.screen_csv}")

    sub["terminal_GlioB"] = sub["RGC_to_GlioB"] + sub["NeuB_to_GlioB"] + sub["GlioB_to_GlioB"]
    sub["terminal_GlioB_fraction"] = sub["terminal_GlioB"] / sub["n_source_cells"]
    dose_order = [0, 2, 4, 6, -2, -4, -6]
    dose_labels = ["Control", "+2", "+4", "+6", "-2", "-4", "-6"]
    blue_shades = ["#b7cde2", "#7fa7c9", CELL_COLORS["GlioB"]]
    orange_shades = ["#f8c997", "#f28e2b", "#a95f16"]
    colors = ["#d7d7d7", *blue_shades, *orange_shades]
    edge_colors = ["#555555", "#7a9cba", "#5b83a7", CELL_COLORS["GlioB"], "#d49b62", CELL_COLORS["NeuB"], "#8b4d12"]

    lookup = {int(round(float(row["dose"]))): row for _, row in sub.iterrows()}
    missing = [d for d in dose_order if d not in lookup]
    if missing:
        raise ValueError(f"Missing {gene} doses: {missing}")

    control = lookup[0]
    n0 = int(control["n_source_cells"])
    x0 = int(control["terminal_GlioB"])
    p0 = float(control["terminal_GlioB_fraction"])

    rng = np.random.default_rng(19491001)
    boot_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    box_values: list[np.ndarray] = []
    pvals: list[float] = []
    for pos, dose in enumerate(dose_order):
        row = lookup[dose]
        n = int(row["n_source_cells"])
        x = int(row["terminal_GlioB"])
        p = float(row["terminal_GlioB_fraction"])
        boot = rng.binomial(n, p, size=28) / n
        box_values.append(boot)
        pooled = (x + x0) / (n + n0)
        denom = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / n + 1.0 / n0), 1e-12))
        z_score = (p - p0) / denom
        pvalue = math.erfc(abs(z_score) / math.sqrt(2.0))
        pvals.append(pvalue if dose != 0 else 1.0)
        for rep, frac in enumerate(boot):
            boot_rows.append(
                {
                    "gene": gene,
                    "dose": dose,
                    "display_label": dose_labels[pos],
                    "bootstrap_replicate": rep,
                    "terminal_GlioB_fraction_bootstrap": frac,
                    "n_source_cells": n,
                    "terminal_GlioB_count": x,
                    "terminal_GlioB_fraction_observed": p,
                }
            )
        summary_rows.append(
            {
                "gene": gene,
                "dose": dose,
                "display_label": dose_labels[pos],
                "n_source_cells": n,
                "terminal_GlioB_count": x,
                "terminal_GlioB_fraction": p,
                "terminal_RGC_count": int(row["RGC_to_RGC"] + row["NeuB_to_RGC"] + row["GlioB_to_RGC"]),
                "terminal_NeuB_count": int(row["RGC_to_NeuB"] + row["NeuB_to_NeuB"] + row["GlioB_to_NeuB"]),
                "NeuB_to_GlioB": int(row["NeuB_to_GlioB"]),
                "GlioB_to_NeuB": int(row["GlioB_to_NeuB"]),
                "two_proportion_z_vs_control": z_score,
                "two_sided_p_vs_control": pvalue if dose != 0 else 1.0,
            }
        )

    # Benjamini-Hochberg q-values for the six non-control dose comparisons.
    nonzero = np.array([i for i, dose in enumerate(dose_order) if dose != 0], dtype=int)
    sorted_nonzero = nonzero[np.argsort(np.asarray(pvals)[nonzero])]
    qvals = np.ones(len(dose_order), dtype=float)
    running = 1.0
    m = float(len(sorted_nonzero))
    for rank_from_end, idx in enumerate(sorted_nonzero[::-1], start=1):
        rank = m - rank_from_end + 1.0
        running = min(running, pvals[idx] * m / rank)
        qvals[idx] = running
    for i, qvalue in enumerate(qvals):
        summary_rows[i]["bh_q_vs_control"] = float(qvalue)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(boot_rows).to_csv(out_dir / "04_Efna5_terminal_GlioB_fraction_bootstrap_source_data.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "04_Efna5_terminal_GlioB_fraction_summary.csv", index=False)

    fig = plt.figure(figsize=(3.35, 4.05))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.15, 1.0], hspace=0.02, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0], sharex=ax)
    fig.subplots_adjust(left=0.18, right=0.975, top=0.86, bottom=0.13)

    positions = np.arange(len(dose_order), dtype=float)
    ax.axvspan(-0.55, 0.55, color="#efefef", zorder=0)
    bp = ax.boxplot(
        box_values,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 0.9},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_edgecolor(edge_colors[i])
        patch.set_alpha(0.76)
        patch.set_linewidth(0.9)
    for element in ["whiskers", "caps"]:
        for i, artist in enumerate(bp[element]):
            artist.set_color(edge_colors[i // 2])

    for i, vals in enumerate(box_values):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), positions[i]) + jitter,
            vals,
            s=8,
            color=edge_colors[i],
            edgecolors="white",
            linewidths=0.25,
            alpha=0.72,
            zorder=3,
        )
        observed = summary_rows[i]["terminal_GlioB_fraction"]
        ax.scatter(
            positions[i],
            observed,
            s=21,
            marker="D",
            color=edge_colors[i],
            edgecolors="white",
            linewidths=0.45,
            zorder=4,
        )
        if dose_order[i] != 0 and qvals[i] < 0.05:
            ax.text(positions[i], 0.585, "*", ha="center", va="center", fontsize=8.5, color=TEXT, weight="bold")

    ax.set_ylim(0.0, 0.60)
    ax.set_xlim(-0.65, len(dose_order) - 0.35)
    ax.set_ylabel("Terminal GlioB fraction")
    ax.set_title("Efna5 perturbation", fontsize=8.2, weight="bold", pad=5)
    ax.grid(axis="y", color="#dedede", lw=0.45, alpha=0.75)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", labelsize=6.2, length=2.5)
    ax.spines["bottom"].set_visible(False)

    ax_bar.axhline(0, color=TEXT, lw=0.85)
    ax_bar.axvspan(-0.55, 0.55, color="#efefef", zorder=0)
    for i, dose in enumerate(dose_order):
        if dose == 0:
            ax_bar.text(positions[i], -0.75, "Control", ha="center", va="top", fontsize=6.2)
            continue
        ax_bar.bar(positions[i], dose, width=0.58, color=colors[i], edgecolor=edge_colors[i], linewidth=0.65)
        va = "bottom" if dose > 0 else "top"
        y = dose + (0.45 if dose > 0 else -0.45)
        ax_bar.text(positions[i], y, f"{dose:+d}", ha="center", va=va, fontsize=6.2, weight="bold")
    ax_bar.set_ylim(-8.0, 7.5)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([])
    ax_bar.set_xlabel("signed perturbation dose", labelpad=9)
    for spine in ["left", "right", "top"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.tick_params(length=0)
    return save_all(fig, args.out_dir / "04_gene_perturbation_dose_response", int(args.dpi))


def make_all_source_shift(args: argparse.Namespace) -> dict[str, str]:
    counts = pd.read_csv(args.counts_csv.expanduser().resolve())
    labels = ["RGC", "NeuB", "GlioB"]
    control = counts.groupby("baseline_terminal_label")["n"].sum().reindex(labels, fill_value=0)
    perturbed = counts.groupby("perturbed_terminal_label")["n"].sum().reindex(labels, fill_value=0)
    n = int(control.sum())
    data = pd.DataFrame({"Control": control / n * 100.0, "Efna5 +6": perturbed / n * 100.0}).T
    switched = int(
        counts.loc[
            (counts["baseline_terminal_label"].astype(str) == "NeuB")
            & (counts["perturbed_terminal_label"].astype(str) == "GlioB"),
            "n",
        ].sum()
    )

    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.78, bottom=0.22)
    y = np.arange(data.shape[0])
    left = np.zeros(data.shape[0], dtype=float)
    for label in labels:
        vals = data[label].to_numpy(float)
        ax.barh(y, vals, left=left, height=0.48, color=CELL_COLORS[label], alpha=0.9, label=label)
        for i, val in enumerate(vals):
            if val >= 8:
                ax.text(left[i] + val / 2, i, f"{val:.0f}%", ha="center", va="center", fontsize=5.8, color="white", weight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(data.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("terminal fate proportion (%)")
    ax.set_title("All-source terminal fate shift", fontsize=8.3, weight="bold", pad=4)
    ax.text(0, 1.32, f"NeuB -> GlioB strict switches: {switched}/{n}", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), ncol=3, fontsize=5.8)
    ax.tick_params(axis="both", labelsize=6.2, length=2.5)
    return save_all(fig, args.out_dir / "04b_Efna5_all_source_terminal_fate_shift", int(args.dpi))


def rounded_node(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str, ec: str = "#333333") -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.8,
        alpha=0.96,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=TEXT, weight="bold")


def arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = "#444444", style: str = "solid") -> None:
    ls = (0, (3.0, 2.0)) if style == "dashed" else "solid"
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.0, linestyle=ls, color=color))


def make_mechanism(args: argparse.Namespace) -> dict[str, str]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    screen = pd.read_csv(args.screen_csv.expanduser().resolve()).copy()
    contrast_path = args.out_dir / "03c_NeuB_GlioB_candidate_gene_contrast_source_data.csv"
    marker_stats = pd.read_csv(contrast_path) if contrast_path.exists() else pd.DataFrame()
    marker_lookup = {}
    if not marker_stats.empty:
        marker_lookup = {str(row["gene"]): row for _, row in marker_stats.iterrows()}

    def row_for(gene: str, dose: float) -> pd.Series:
        sub = screen[(screen["gene"].astype(str) == gene) & np.isclose(screen["dose"].astype(float), float(dose))]
        if sub.empty:
            raise ValueError(f"Missing perturbation row for {gene} dose {dose}")
        return sub.iloc[0]

    def terminal_count(row: pd.Series, fate: str) -> int:
        return int(row[f"RGC_to_{fate}"] + row[f"NeuB_to_{fate}"] + row[f"GlioB_to_{fate}"])

    def marker_summary(gene: str) -> tuple[float, str, bool]:
        if gene not in marker_lookup:
            return float("nan"), "not available", False
        row = marker_lookup[gene]
        log2fc = float(row["log2FC_GlioB_vs_NeuB"])
        if log2fc >= 0.3:
            return log2fc, "GlioB-enriched", True
        if log2fc <= -0.3:
            return log2fc, "NeuB-enriched", True
        return log2fc, "weak NeuB/GlioB contrast", False

    candidates = [
        {
            "gene": "Efna5",
            "class": "primary driver candidate",
            "dose": 6.0,
            "target_fate": "GlioB",
            "switch_event": "NeuB_to_GlioB",
            "kegg_id": "mmu:13640",
            "kegg_pathway": "KEGG direct: mmu04360 Axon guidance; mmu04151 PI3K-Akt; mmu04010 MAPK; mmu04014 Ras; mmu04015 Rap1",
            "kegg_mapper_direct_hit": True,
            "upstream_signal": "Eph/ephrin contact cues",
            "downstream_module": "EphA receptors; migration and progenitor-state control",
            "module": "EFNA5-EphA",
            "color": OE_RED,
            "edge_color": OE_RED,
            "edge_rad": -0.12,
            "x": 0.250,
            "y": 0.485,
            "w": 0.165,
            "h": 0.160,
        },
        {
            "gene": "Dbi",
            "class": "supporting GlioB-shift candidate",
            "dose": 6.0,
            "target_fate": "GlioB",
            "switch_event": "NeuB_to_GlioB",
            "kegg_id": "mmu:13167",
            "kegg_pathway": "KEGG direct: mmu03320 PPAR signaling; literature: DBI-GABA(A) neurogenesis module",
            "kegg_mapper_direct_hit": True,
            "upstream_signal": "radial glia/intermediate progenitor state",
            "downstream_module": "GABA(A) modulation; neurogenesis balance",
            "module": "DBI-GABA(A)",
            "color": OE_RED,
            "edge_color": "#b2182b",
            "edge_rad": -0.06,
            "x": 0.230,
            "y": 0.690,
            "w": 0.145,
            "h": 0.145,
        },
        {
            "gene": "Adgrb3",
            "class": "supporting GlioB-shift candidate",
            "dose": 6.0,
            "target_fate": "GlioB",
            "switch_event": "NeuB_to_GlioB",
            "kegg_id": "mmu:210933",
            "kegg_pathway": "No direct KEGG Mapper hit; literature: BAI3-ELMO1-Rac1 module",
            "kegg_mapper_direct_hit": False,
            "upstream_signal": "adhesion GPCR / BAI3 signaling",
            "downstream_module": "ELMO1-Rac1; cytoskeleton and morphogenesis",
            "module": "ADGRB3/BAI3",
            "color": OE_RED,
            "edge_color": "#c94741",
            "edge_rad": 0.04,
            "x": 0.435,
            "y": 0.690,
            "w": 0.145,
            "h": 0.145,
        },
        {
            "gene": "Ptn",
            "class": "opposite-direction candidate",
            "dose": 6.0,
            "target_fate": "NeuB",
            "switch_event": "GlioB_to_NeuB",
            "kegg_id": "mmu:19242",
            "kegg_pathway": "No direct KEGG Mapper hit; literature: PTN-ALK/PTPRZ PI3K/MAPK module",
            "kegg_mapper_direct_hit": False,
            "upstream_signal": "PTN ligand module",
            "downstream_module": "ALK/PTPRZ; PI3K-Akt/MAPK; neurite growth",
            "module": "PTN-ALK/PTPRZ",
            "color": OE_RED,
            "edge_color": "#ef8a62",
            "edge_rad": 0.08,
            "x": 0.250,
            "y": 0.230,
            "w": 0.145,
            "h": 0.145,
        },
        {
            "gene": "Fzd3",
            "class": "opposite-direction candidate",
            "dose": 6.0,
            "target_fate": "NeuB",
            "switch_event": "GlioB_to_NeuB",
            "kegg_id": "mmu:14365",
            "kegg_pathway": "KEGG direct: mmu04310 Wnt signaling and mmu04360 Axon guidance; literature: Wnt7-Celsr3-Fzd3-Jag1/Notch module",
            "kegg_mapper_direct_hit": True,
            "upstream_signal": "Wnt/Celsr polarity signal",
            "downstream_module": "Jag1/Notch timing; neurogenic-gliogenic balance",
            "module": "FZD3-Wnt/Notch",
            "color": OE_RED,
            "edge_color": "#f4a582",
            "edge_rad": -0.02,
            "x": 0.435,
            "y": 0.230,
            "w": 0.145,
            "h": 0.145,
        },
    ]

    control_row = row_for("Efna5", 0.0)
    control_glio = terminal_count(control_row, "GlioB") / float(control_row["n_source_cells"])
    control_neub = terminal_count(control_row, "NeuB") / float(control_row["n_source_cells"])
    candidate_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for cand in candidates:
        row = row_for(cand["gene"], cand["dose"])
        n = int(row["n_source_cells"])
        target = str(cand["target_fate"])
        target_fraction = terminal_count(row, target) / float(n)
        control_fraction = control_glio if target == "GlioB" else control_neub
        strict_count = int(row[str(cand["switch_event"])])
        strict_rate = strict_count / float(n)
        log2fc, marker_direction, marker_evidence = marker_summary(str(cand["gene"]))
        perturbation_evidence = strict_count > 50
        kegg_literature_evidence = True
        context_evidence_type = "KEGG/literature-supported" if bool(cand["kegg_mapper_direct_hit"]) else "literature-supported"
        evidence_count = int(perturbation_evidence) + int(marker_evidence) + int(kegg_literature_evidence)
        candidate_rows.append(
            {
                "gene": cand["gene"],
                "class": cand["class"],
                "kegg_id": cand["kegg_id"],
                "selected_dose": cand["dose"],
                "n_source_cells": n,
                "perturbation_direction": f"{cand['switch_event']} / terminal {target}",
                "strict_switch_count": strict_count,
                "strict_switch_rate": strict_rate,
                "target_terminal_fraction_control": control_fraction,
                "target_terminal_fraction_perturbed": target_fraction,
                "target_terminal_fraction_delta": target_fraction - control_fraction,
                "marker_log2FC_GlioB_vs_NeuB": log2fc,
                "marker_enrichment": marker_direction,
                "kegg_pathway_or_module": cand["kegg_pathway"],
                "upstream_signal": cand["upstream_signal"],
                "downstream_gene_or_process": cand["downstream_module"],
                "perturbation_evidence": perturbation_evidence,
                "marker_evidence": marker_evidence,
                "kegg_mapper_direct_hit": bool(cand["kegg_mapper_direct_hit"]),
                "context_evidence_type": context_evidence_type,
                "kegg_literature_evidence": kegg_literature_evidence,
                "evidence_count": evidence_count,
                "included_in_main_figure": evidence_count >= 2,
            }
        )
        edge_rows.extend(
            [
                {
                    "source": cand["gene"],
                    "target": target,
                    "edge_type": "model-inferred",
                    "evidence": f"{cand['dose']:+.0f} perturbation; {strict_count}/{n} strict switches",
                    "direction": "positive",
                    "line_style": "dashed",
                    "line_width_metric": strict_rate,
                },
                {
                    "source": cand["upstream_signal"],
                    "target": cand["gene"],
                    "edge_type": context_evidence_type,
                    "evidence": cand["kegg_pathway"],
                    "direction": "context",
                    "line_style": "solid",
                    "line_width_metric": np.nan,
                },
                {
                    "source": cand["gene"],
                    "target": cand["downstream_module"],
                    "edge_type": context_evidence_type,
                    "evidence": cand["module"],
                    "direction": "positive/context",
                    "line_style": "solid",
                    "line_width_metric": np.nan,
                },
            ]
        )

    evidence_rows = [
        {
            "ref": "[1]",
            "module": "STORIES / potential landscape",
            "process": "Potential-ordered trajectories can nominate fate-associated genes and regulators.",
            "figure_use": "Analysis logic for ordering cells and reading branch programs.",
            "citation": "Huizing et al., STORIES: learning cell fate landscapes from spatial transcriptomics using optimal transport, Nature Methods, 2025.",
            "url": "https://www.nature.com/articles/s41592-025-02855-4",
        },
        {
            "ref": "[2]",
            "module": "KEGG Mapper Color",
            "process": "KEGG identifiers can be colored on pathway maps with background/foreground colors; Chrome Mapper run directly hit Efna5, Dbi, and Fzd3 but not Ptn or Adgrb3.",
            "figure_use": "Used only for pathway context; not treated as direct perturbation validation.",
            "citation": "KEGG Mapper Color documentation.",
            "url": "https://www.genome.jp/kegg/mapper/color.html",
        },
        {
            "ref": "[3]",
            "module": "EFNA5-EphA",
            "process": "Ephrin-A/EphA signaling regulates neural progenitor apoptosis and early brain morphology.",
            "figure_use": "Supports EFNA5 as a contact-dependent developmental signal near progenitor fate decisions.",
            "citation": "Depaepe et al., Nature, 2005; Park et al., Cell Death Differentiation, 2013.",
            "url": "https://www.nature.com/articles/nature03651; https://pmc.ncbi.nlm.nih.gov/articles/PMC3524646/",
        },
        {
            "ref": "[4]",
            "module": "DBI-GABA(A)",
            "process": "DBI is expressed in radial glia/intermediate progenitors and modulates GABA-dependent embryonic neurogenesis.",
            "figure_use": "Supports DBI as a progenitor-state and neurogenesis-balance candidate.",
            "citation": "Everlien et al., Neuron, 2022.",
            "url": "https://www.sciencedirect.com/science/article/pii/S0896627322006675",
        },
        {
            "ref": "[5]",
            "module": "PTN-ALK/PTPRZ and FZD3-Wnt/Notch",
            "process": "PTN and Wnt/Fzd modules provide pathway context for neural differentiation and fate timing.",
            "figure_use": "Supports opposite-direction and fate-timing candidate modules.",
            "citation": "Karsten et al., J Neurosci, 2003; Wang et al., Nature Communications, 2016.",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6729244/; https://www.nature.com/articles/ncomms10936",
        },
        {
            "ref": "[6]",
            "module": "ADGRB3/BAI3-ELMO1-Rac1",
            "process": "BAI3/ADGRB3 regulates neuronal dendrite morphogenesis through ELMO1 and Rac1.",
            "figure_use": "Supports ADGRB3 as an adhesion/cytoskeleton developmental module.",
            "citation": "Lanoue et al., Molecular Psychiatry, 2013.",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3730300/",
        },
    ]
    pd.DataFrame(evidence_rows).to_csv(args.out_dir / "05_mechanism_literature_evidence.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(args.out_dir / "05_candidate_gene_evidence_table.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(args.out_dir / "05_mechanism_edge_table.csv", index=False)
    kegg_dir = args.out_dir / "kegg_mapper"
    kegg_dir.mkdir(parents=True, exist_ok=True)
    kegg_lines = []
    for row in candidate_rows:
        foreground = DE_BLUE if bool(row["marker_evidence"]) else "#ffffff"
        kegg_lines.append(f"{row['kegg_id']} {OE_RED},{foreground}")
    (kegg_dir / "mouse_midbrain_oe_red_de_blue_color_input.txt").write_text("\n".join(kegg_lines) + "\n", encoding="utf-8")

    def mini_fate_bar(ax: plt.Axes, x: float, y: float, w: float, h: float, values: dict[str, float], title: str) -> None:
        ax.text(x, y + h + 0.012, title, ha="left", va="bottom", fontsize=5.6, color=MUTED, weight="bold")
        left = x
        for label in ["RGC", "NeuB", "GlioB"]:
            ww = w * float(values[label]) / 100.0
            ax.add_patch(Rectangle((left, y), ww, h, facecolor=CELL_COLORS[label], edgecolor="white", linewidth=0.35))
            left += ww
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#555555", linewidth=0.45))
        ax.text(x + w + 0.007, y + h / 2, f"GlioB {values['GlioB']:.0f}%", ha="left", va="center", fontsize=5.15, color=CELL_COLORS["GlioB"], weight="bold")

    def card(
        ax: plt.Axes,
        xy: tuple[float, float],
        wh: tuple[float, float],
        title: str,
        lines: list[str],
        fc: str,
        ec: str,
        *,
        main: bool = False,
        evidence: tuple[bool, bool, str] = (True, True, "K/L"),
    ) -> None:
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.026",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.75,
            alpha=0.98,
        )
        ax.add_patch(patch)
        ax.add_patch(Rectangle((x + 0.008, y + h - 0.018), w - 0.016, 0.009, facecolor=OE_RED, edgecolor="none", alpha=0.84))
        ax.text(x + 0.014, y + h - 0.020, title, ha="left", va="top", fontsize=7.1 if main else 6.05, color=TEXT, weight="bold", fontstyle="italic")
        for i, line in enumerate(lines):
            ax.text(x + 0.014, y + h - 0.060 - 0.024 * i, line, ha="left", va="top", fontsize=4.75 if main else 4.45, color=MUTED)
        chip_y = y + 0.010
        perturb_ok, marker_ok, context_label = evidence
        for i, (label, ok) in enumerate([("P", perturb_ok), ("M", marker_ok), (context_label, True)]):
            chip_fc = {"P": "#fee0d2", "M": "#deebf7", "K/L": "#ffffff"}.get(label, "#ffffff")
            chip_ec = {"P": OE_RED, "M": DE_BLUE, "K/L": "#777777"}.get(label, "#777777")
            if label not in {"P", "M", "K/L"}:
                chip_fc = "#ffffff"
                chip_ec = "#777777"
            if not ok:
                chip_fc = "#f3f3f3"
                chip_ec = "#b8b8b8"
            ax.add_patch(FancyBboxPatch((x + 0.014 + 0.036 * i, chip_y), 0.028, 0.024, boxstyle="round,pad=0.004,rounding_size=0.006", facecolor=chip_fc, edgecolor=chip_ec, linewidth=0.45))
            ax.text(x + 0.028 + 0.036 * i, chip_y + 0.012, label, ha="center", va="center", fontsize=3.85, color=chip_ec, weight="bold")

    def node_center(cand: dict[str, Any]) -> tuple[float, float]:
        return float(cand["x"]) + float(cand["w"]) / 2.0, float(cand["y"]) + float(cand["h"]) / 2.0

    def model_arrow(cand: dict[str, Any], target: tuple[float, float], label: str) -> None:
        x0, y0 = node_center(cand)
        color = str(cand["edge_color"])
        count = next(r for r in candidate_rows if r["gene"] == cand["gene"])["strict_switch_rate"]
        lw = 0.85 + 8.2 * float(count)
        ax.add_patch(
            FancyArrowPatch(
                (x0 + float(cand["w"]) * 0.48, y0),
                target,
                connectionstyle=f"arc3,rad={cand['edge_rad']}",
                arrowstyle="-|>",
                mutation_scale=10,
                lw=lw,
                linestyle=(0, (3, 2)),
                color=color,
                alpha=0.94,
                zorder=2,
            )
        )
        _ = label

    def inset_row(y: float, labels: list[str], color: str) -> None:
        xs = [0.705, 0.795, 0.890]
        widths = [0.064, 0.070, 0.074]
        for i, text in enumerate(labels):
            ax.add_patch(FancyBboxPatch((xs[i], y), widths[i], 0.033, boxstyle="round,pad=0.006,rounding_size=0.008", facecolor="#ffffff", edgecolor=color if i == 0 else "#b6b6b6", linewidth=0.55))
            ax.text(xs[i] + widths[i] / 2, y + 0.0165, text, ha="center", va="center", fontsize=4.15, color=TEXT if i < 2 else MUTED)
            if i < 2:
                ax.add_patch(FancyArrowPatch((xs[i] + widths[i] + 0.004, y + 0.0165), (xs[i + 1] - 0.004, y + 0.0165), arrowstyle="-|>", mutation_scale=6, lw=0.55, color="#777777"))

    fig, ax = plt.subplots(figsize=(8.1, 4.65))
    fig.subplots_adjust(left=0.018, right=0.986, top=0.93, bottom=0.070)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.025, 0.965, "Perturbation-prioritized gene-fate evidence network", ha="left", va="top", fontsize=9.1, weight="bold", color=TEXT)
    ax.text(0.025, 0.930, "Candidate genes pass >=2 evidence classes: perturbation effect, marker expression, and KEGG/literature context.", ha="left", va="top", fontsize=5.65, color=MUTED)

    ax.add_patch(Rectangle((0.025, 0.165), 0.635, 0.710, facecolor="#fbfbfb", edgecolor="#dddddd", linewidth=0.70))
    ax.text(0.040, 0.848, "data-driven candidate network", ha="left", va="top", fontsize=6.4, weight="bold", color=TEXT)
    rounded_node(ax, (0.055, 0.445), (0.128, 0.138), "E12.5 RGC\nhigh potential", "#dff2e7", CELL_COLORS["RGC"])
    ax.text(0.119, 0.425, "source state", ha="center", va="top", fontsize=5.05, color=MUTED)

    for cand in candidates:
        row = next(r for r in candidate_rows if r["gene"] == cand["gene"])
        effect = f"{row['strict_switch_count']}/{row['n_source_cells']} switch ({100 * row['strict_switch_rate']:.1f}%)"
        marker = f"marker {row['marker_log2FC_GlioB_vs_NeuB']:+.2f}" if np.isfinite(row["marker_log2FC_GlioB_vs_NeuB"]) else "marker n/a"
        target = "terminal GlioB" if cand["target_fate"] == "GlioB" else "terminal NeuB"
        delta = f"{target} {100 * row['target_terminal_fraction_control']:.0f}->{100 * row['target_terminal_fraction_perturbed']:.0f}%"
        fc = "#fff5f0"
        context_label = "K" if bool(cand["kegg_mapper_direct_hit"]) else "L"
        card(
            ax,
            (cand["x"], cand["y"]),
            (cand["w"], cand["h"]),
            cand["gene"],
            [delta, effect, marker],
            fc,
            cand["edge_color"],
            main=(cand["gene"] == "Efna5"),
            evidence=(bool(row["perturbation_evidence"]), bool(row["marker_evidence"]), context_label),
        )
        ax.add_patch(FancyArrowPatch((0.183, 0.514), (cand["x"], cand["y"] + cand["h"] * 0.50), arrowstyle="-", lw=0.45, color="#cfcfcf", alpha=0.85, zorder=1))

    rounded_node(ax, (0.555, 0.545), (0.088, 0.105), "GlioB\nfate", "#e8f0fa", CELL_COLORS["GlioB"])
    rounded_node(ax, (0.555, 0.295), (0.088, 0.105), "NeuB\nfate", "#fff0dc", CELL_COLORS["NeuB"])
    glio_target = (0.555, 0.598)
    neub_target = (0.555, 0.348)
    for cand in candidates:
        row = next(r for r in candidate_rows if r["gene"] == cand["gene"])
        label = f"{100 * row['strict_switch_rate']:.1f}%"
        model_arrow(cand, glio_target if cand["target_fate"] == "GlioB" else neub_target, label)

    legend_y = 0.126
    ax.text(
        0.047,
        legend_y,
        "Red = overexpression target; blue evidence chips = differential/marker gene; dashed arrows = model-inferred fate shift; inset = KEGG/literature context.",
        ha="left",
        va="center",
        fontsize=5.0,
        color=MUTED,
    )

    ax.add_patch(Rectangle((0.685, 0.605), 0.290, 0.270, facecolor="#fbfbfb", edgecolor="#d5d5d5", linewidth=0.65))
    ax.text(0.700, 0.850, "all-source readout", ha="left", va="top", fontsize=6.35, weight="bold", color=TEXT)
    mini_fate_bar(ax, 0.705, 0.770, 0.170, 0.030, {"RGC": 45.0, "NeuB": 38.0, "GlioB": 18.0}, "control")
    mini_fate_bar(ax, 0.705, 0.695, 0.170, 0.030, {"RGC": 25.0, "NeuB": 21.0, "GlioB": 54.0}, "Efna5 +6")
    ax.text(0.700, 0.642, "main effect: Efna5 +6, 277/1708 NeuB -> GlioB", ha="left", va="bottom", fontsize=5.25, color=CELL_COLORS["GlioB"], weight="bold")

    ax.add_patch(Rectangle((0.685, 0.245), 0.290, 0.315, facecolor="#fbfbfb", edgecolor="#d5d5d5", linewidth=0.65))
    ax.text(
        0.700,
        0.538,
        "simplified KEGG/literature inset",
        ha="left",
        va="top",
        fontsize=6.35,
        weight="bold",
        color=TEXT,
        bbox={"facecolor": "#fbfbfb", "edgecolor": "none", "pad": 0.4},
        zorder=6,
    )
    inset_row(0.488, ["Efna5", "EphA", "contact/migration"], OE_RED)
    inset_row(0.438, ["Dbi", "GABA(A)", "neurogenesis"], OE_RED)
    inset_row(0.388, ["Adgrb3", "ELMO/Rac1", "cytoskeleton"], OE_RED)
    inset_row(0.338, ["Ptn", "ALK/PTPRZ", "PI3K/MAPK"], OE_RED)
    inset_row(0.288, ["Fzd3", "Wnt/Notch", "fate timing"], OE_RED)

    ax.add_patch(Rectangle((0.685, 0.165), 0.290, 0.052, facecolor="#ffffff", edgecolor="#e3e3e3", linewidth=0.55))
    ax.text(0.700, 0.194, "Red marks overexpression targets; blue foreground marks differential genes.", ha="left", va="center", fontsize=4.95, color=MUTED)
    ax.text(0.025, 0.050, "Claim boundary: dashed gene-to-fate arrows are computational perturbation readouts; solid inset edges summarize KEGG/literature context, not direct experimental validation.", ha="left", va="bottom", fontsize=5.2, color=MUTED)
    return save_all(fig, args.out_dir / "05_proposed_perturbation_mechanism", int(args.dpi))


def make_contact_sheet(args: argparse.Namespace, outputs: dict[str, dict[str, str]]) -> dict[str, str]:
    keys = [
        ("01 screen", outputs["screen"]["png"]),
        ("02 trajectory", outputs["trajectory"]["png"]),
        ("03 expression", outputs["expression_rgc_gliob_stories"]["png"]),
        ("04 dose", outputs["dose"]["png"]),
        ("04b fate shift", outputs["fate_shift"]["png"]),
        ("05 mechanism", outputs["mechanism"]["png"]),
    ]
    fig = plt.figure(figsize=(8.6, 8.2))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.23, wspace=0.08)
    for i, (label, path) in enumerate(keys):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.01, 0.99, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.0, weight="bold", color=TEXT)
    return save_all(fig, args.out_dir / "00_final_perturbation_story_contact_sheet", int(args.dpi), tight=True)


def main() -> None:
    args = parse_args()
    setup_style()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phate = pd.read_csv(args.phate_csv.expanduser().resolve())
    if "label" not in phate.columns and "cell_type" in phate.columns:
        phate["label"] = phate["cell_type"].astype(str)

    outputs: dict[str, dict[str, str]] = {}
    outputs["screen"] = make_screen_heatmap(args)
    outputs["trajectory"] = make_trajectory_pair(args, phate)
    outputs["expression"] = make_expression_heatmap(args, phate)
    outputs["expression_two_type"] = make_two_type_expression_heatmap(args, phate)
    outputs["expression_two_type_contrast"] = make_two_type_expression_contrast(args)
    outputs["expression_two_fate_stories"] = make_two_fate_stories_heatmap(args, phate)
    outputs["expression_rgc_gliob_stories"] = make_rgc_branch_stories_heatmap(args, phate, "GlioB")
    outputs["expression_rgc_neub_stories"] = make_rgc_branch_stories_heatmap(args, phate, "NeuB")
    outputs["dose"] = make_dose_response(args)
    outputs["fate_shift"] = make_all_source_shift(args)
    outputs["mechanism"] = make_mechanism(args)
    outputs["contact_sheet"] = make_contact_sheet(args, outputs)

    summary = {
        "status": "DONE",
        "definition": "Final draft perturbation story figures for the mouse dorsal-midbrain SpaPOT result.",
        "source_csv": str(args.source_csv.expanduser().resolve()),
        "counts_csv": str(args.counts_csv.expanduser().resolve()),
        "screen_csv": str(args.screen_csv.expanduser().resolve()),
        "outputs": outputs,
    }
    summary_path = args.out_dir / "final_perturbation_story_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

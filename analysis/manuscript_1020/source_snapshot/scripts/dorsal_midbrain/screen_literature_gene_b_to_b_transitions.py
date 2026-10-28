#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import make_gene_perturbation_paired_trajectory_panel as paired  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RUN = paired.ACTIVE_RUN
DEFAULT_OUT_DIR = PROJECT_ROOT / "result_figures" / "小鼠中脑" / "g扰动" / "literature_gene_b_to_b_screen"
DEFAULT_GAT_CHECKPOINT = paired.DEFAULT_GAT_CHECKPOINT

CELL_TYPES = ["RGC", "NeuB", "GlioB"]
CELL_COLORS = paired.CELL_COLORS

LITERATURE_CANDIDATES: list[dict[str, str]] = [
    {"gene": "Efna5", "role": "axon-guidance / midbrain-patterning marker", "basis": "Eph/ephrin signaling and MOSTA marker"},
    {"gene": "Dbx1", "role": "dorsal midbrain progenitor / patterning", "basis": "dorsal midbrain and MOSTA marker"},
    {"gene": "Adgrb3", "role": "neural adhesion receptor / neuroblast marker", "basis": "MOSTA neuroblast marker and neural receptor"},
    {"gene": "Neurod1", "role": "neurogenic differentiation", "basis": "proneural differentiation factor"},
    {"gene": "Fabp7", "role": "radial glia / neural stem cell", "basis": "radial glia marker and functional regulator"},
    {"gene": "Ptn", "role": "neuroglial progenitor growth factor", "basis": "developmental neuroglial progenitor expression"},
    {"gene": "Mdk", "role": "developmental neurotrophic growth factor", "basis": "midkine family neural development"},
    {"gene": "Vim", "role": "radial glia / progenitor cytoskeleton", "basis": "radial glia and neural progenitor marker"},
    {"gene": "Dbi", "role": "radial glia / gliogenic candidate", "basis": "screen-ranked radial-glial-associated gene"},
    {"gene": "Id2", "role": "glial differentiation regulator", "basis": "Id family regulation of oligodendrocyte/glial differentiation"},
    {"gene": "Id3", "role": "NSPC-to-astrocyte differentiation", "basis": "BMP/Id3 astrocytic differentiation evidence"},
    {"gene": "Sox11", "role": "neuronal differentiation / neurite growth", "basis": "neural differentiation and neuron survival"},
    {"gene": "Cdkn1c", "role": "neurogenic transition / cell-cycle exit", "basis": "progenitor-to-neurogenic division transition"},
    {"gene": "Chd7", "role": "neural progenitor differentiation", "basis": "chromatin regulator of neural progenitor differentiation"},
    {"gene": "Rest", "role": "neuronal gene repression", "basis": "neural fate transcriptional repression"},
    {"gene": "Nog", "role": "BMP antagonist / neural induction", "basis": "BMP pathway neurodevelopment candidate"},
    {"gene": "Smad5", "role": "BMP pathway / glial-neural balance", "basis": "BMP pathway neurodevelopment candidate"},
    {"gene": "Cdc42", "role": "radial glial cytoskeleton / polarity", "basis": "radial glia scaffold and actin regulation"},
    {"gene": "Fzd3", "role": "Wnt/frizzled neuroblast development", "basis": "neurogenesis and target innervation"},
    {"gene": "Zic4", "role": "neural patterning", "basis": "developmental transcription factor candidate"},
    {"gene": "Sox6", "role": "neuronal lineage maturation", "basis": "neural transcription factor candidate"},
    {"gene": "Erbb4", "role": "neuregulin signaling / neuroblast development", "basis": "neural development signaling candidate"},
    {"gene": "Tal2", "role": "midbrain/diencephalon developmental TF", "basis": "neural bHLH patterning candidate"},
    {"gene": "St18", "role": "glial transcription factor candidate", "basis": "glial lineage transcription candidate"},
    {"gene": "Nedd4", "role": "neural progenitor signaling", "basis": "screen-ranked developmental signaling candidate"},
    {"gene": "Pax6", "role": "radial glia / neural progenitor", "basis": "canonical RGC progenitor factor"},
    {"gene": "Sox2", "role": "radial glia / neural stem cell", "basis": "canonical neural stem/progenitor factor"},
    {"gene": "Ascl1", "role": "proneural progenitor", "basis": "canonical proneural bHLH factor"},
    {"gene": "Neurog2", "role": "proneural progenitor", "basis": "canonical proneural bHLH factor"},
    {"gene": "Gfap", "role": "astroglial marker", "basis": "canonical astroglial marker"},
    {"gene": "Olig2", "role": "oligodendrocyte / glial lineage", "basis": "canonical Olig lineage factor"},
    {"gene": "Aldh1l1", "role": "astroglial / glioblast marker", "basis": "STORIES/MOSTA glial marker"},
    {"gene": "Mki67", "role": "proliferation", "basis": "proliferating progenitor marker"},
    {"gene": "Npas3", "role": "neuroblast marker", "basis": "MOSTA neuroblast marker"},
    {"gene": "Hes1", "role": "glioblast regulon", "basis": "MOSTA dorsal midbrain regulon"},
    {"gene": "Hes6", "role": "neuroblast regulon", "basis": "MOSTA dorsal midbrain regulon"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch screen literature-supported dorsal-midbrain candidate genes for strict B-to-B terminal switching."
    )
    parser.add_argument("--run-dir", type=Path, default=ACTIVE_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_GAT_CHECKPOINT)
    parser.add_argument("--genes", nargs="*", default=None)
    parser.add_argument("--doses", nargs="*", type=float, default=[-4.0, 4.0])
    parser.add_argument("--source-time", type=float, default=12.5)
    parser.add_argument("--terminal-time", type=float, default=16.5)
    parser.add_argument("--source-label", default="RGC")
    parser.add_argument("--max-source-cells", type=int, default=384)
    parser.add_argument("--seed", type=int, default=19491001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--metric-dt", type=float, default=0.5)
    parser.add_argument("--ode-chunk-size", type=int, default=256)
    parser.add_argument("--gat-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=2048)
    parser.add_argument("--knn-k", type=int, default=31)
    parser.add_argument("--expression-clip-quantile", type=float, default=0.999)
    parser.add_argument("--latent-add-x", type=float, default=0.32)
    parser.add_argument("--latent-add-max-x", type=float, default=0.64)
    parser.add_argument("--latent-direction-expression-dose", type=float, default=1.0)
    parser.add_argument("--fig-width", type=float, default=5.7)
    parser.add_argument("--fig-height", type=float, default=4.0)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
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


def candidate_table(genes: list[str] | None, model_genes: set[str]) -> pd.DataFrame:
    info = pd.DataFrame(LITERATURE_CANDIDATES)
    if genes:
        requested = pd.DataFrame({"gene": list(dict.fromkeys(map(str, genes)))})
        info = requested.merge(info, on="gene", how="left")
        info["role"] = info["role"].fillna("user-requested candidate")
        info["basis"] = info["basis"].fillna("explicit input")
    info["in_current_2000_gene_model"] = info["gene"].astype(str).isin(model_genes)
    info["screened_now"] = info["in_current_2000_gene_model"]
    return info


def transition_counts(base_labels: np.ndarray, pert_labels: np.ndarray) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for src in CELL_TYPES:
        for dst in CELL_TYPES:
            n = int(np.sum((base_labels == src) & (pert_labels == dst)))
            counts[f"{src}_to_{dst}"] = n
            rows.append({"baseline_terminal_label": src, "perturbed_terminal_label": dst, "n": n})
    return counts, rows


def plot_ranked_screen(df: pd.DataFrame, candidate_df: pd.DataFrame, out_dir: Path, width: float, height: float) -> dict[str, str]:
    cur = df.copy()
    cur["condition"] = cur["gene"].astype(str) + " " + np.where(cur["dose"] > 0, "+", "") + cur["dose"].map(lambda v: f"{v:g}")
    cur["max_b_to_b"] = cur[["NeuB_to_GlioB", "GlioB_to_NeuB"]].max(axis=1)
    keep = cur.sort_values(["max_b_to_b", "NeuB_to_GlioB", "GlioB_to_NeuB"], ascending=False).head(22).copy()
    keep = keep.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(width, height))
    y = np.arange(len(keep))
    ax.barh(y + 0.18, keep["NeuB_to_GlioB"], height=0.32, color=CELL_COLORS["GlioB"], alpha=0.88, label="NeuB -> GlioB")
    ax.barh(y - 0.18, keep["GlioB_to_NeuB"], height=0.32, color=CELL_COLORS["NeuB"], alpha=0.88, label="GlioB -> NeuB")
    ax.set_yticks(y)
    ax.set_yticklabels(keep["condition"])
    ax.set_xlabel("strict B-to-B terminal switches among 384 E12.5 RGC-origin cells")
    ax.set_title("Literature-supported gene perturbation screen")
    ax.legend(loc="lower right", fontsize=6.2)
    ax.grid(axis="x", color="#dddddd", lw=0.45, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, row in keep.iterrows():
        m = max(float(row["NeuB_to_GlioB"]), float(row["GlioB_to_NeuB"]))
        if m > 0:
            ax.text(m + 0.35, i, f"{int(m)}", va="center", ha="left", fontsize=5.8, color="#333333")
    return save_all(fig, out_dir / "literature_gene_b_to_b_ranked_switches")


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    rows = []
    for gene, sub in df.groupby("gene", sort=False):
        row = {"gene": gene}
        for _, r in sub.iterrows():
            dose_label = "plus4" if float(r["dose"]) > 0 else "minus4"
            row[f"{dose_label}_NeuB_to_GlioB"] = float(r["NeuB_to_GlioB"])
            row[f"{dose_label}_GlioB_to_NeuB"] = float(r["GlioB_to_NeuB"])
        rows.append(row)
    mat = pd.DataFrame(rows).fillna(0.0)
    value_cols = [c for c in ["minus4_NeuB_to_GlioB", "plus4_NeuB_to_GlioB", "minus4_GlioB_to_NeuB", "plus4_GlioB_to_NeuB"] if c in mat]
    mat["max_switch"] = mat[value_cols].max(axis=1)
    mat = mat.sort_values("max_switch", ascending=False).head(24).reset_index(drop=True)
    values = mat[value_cols].to_numpy(float)

    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    cmap = LinearSegmentedColormap.from_list("white_to_signal", ["#f8f8f8", "#d8e8ef", "#7da9c8", "#2f5d8a"])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=max(1.0, float(np.nanmax(values))))
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels(mat["gene"])
    labels = [c.replace("minus4_", "-4 ").replace("plus4_", "+4 ").replace("_to_", " -> ") for c in value_cols]
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Strict fate-switch counts")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = int(round(values[i, j]))
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=5.8, color="#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("cell count")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return save_all(fig, out_dir / "literature_gene_b_to_b_switch_heatmap")


def make_screen(args: argparse.Namespace) -> dict[str, Any]:
    setup_style()
    paired.perturb.seed_all(int(args.seed))
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = paired.runtime_args(args)
    device = paired.perturb.resolve_device(str(args.device))

    summary = paired.perturb.load_json(run_dir / "summary.json")
    adata = ad.read_h5ad(summary["trained_h5ad"])
    adata.obs[rt.annotation_key] = adata.obs[rt.annotation_key].astype("category")
    expression = paired.dense_float32(adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X)
    expr_stats = paired.perturb.expression_stats(expression, float(args.expression_clip_quantile))
    pairs = paired.perturb.observed_time_pairs(adata, rt)
    source_input = paired.perturb.raw_to_input(float(args.source_time), pairs)
    terminal_input = paired.perturb.raw_to_input(float(args.terminal_time), pairs)
    initial_input = float(np.min(np.asarray(adata.obs[rt.time_input_key], dtype=np.float64)))
    spatial_dim = int(adata.obsm[rt.spatial_key].shape[1])
    method = str(summary.get("config", {}).get("ode_method", "rk4"))
    step_size = float(summary.get("config", {}).get("ode_step_size", 0.25))

    model = paired.perturb.load_dynamics(summary, adata, rt, device)
    classifier, label_map, classifier_path = paired.perturb.load_classifier(run_dir, device)
    gat = paired.perturb.load_gat_bundle(adata, device, args.decoder_checkpoint.expanduser().resolve())
    if list(map(str, adata.var_names)) != gat.gene_names:
        raise ValueError("Trained h5ad var_names do not match the GAT checkpoint gene order.")
    knn = paired.perturb.build_terminal_knn(adata, rt, terminal_input, spatial_dim, int(args.knn_k))

    source_indices = paired.perturb.select_source_indices(adata, rt, initial_input, int(args.seed))
    control = paired.perturb.make_control_pack(
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

    time_input_grid, base_states = paired.record_rollout(
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
    base_probs, base_labels, classes = paired.perturb.classifier_readout(
        classifier,
        label_map,
        torch.as_tensor(base_states[-1], dtype=torch.float32, device=device),
        terminal_input,
        spatial_dim,
        batch_size=int(args.classifier_batch_size),
    )
    model_genes = set(map(str, adata.var_names))
    candidates = candidate_table(args.genes, model_genes)
    screened = candidates[candidates["screened_now"]].copy()
    screened_genes = list(screened["gene"].astype(str))
    gene_to_idx = {str(gene): int(idx) for idx, gene in enumerate(map(str, adata.var_names))}
    class_to_idx = {name: i for i, name in enumerate(classes)}

    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for _, cand in screened.iterrows():
        gene = str(cand["gene"])
        gene_idx = gene_to_idx[gene]
        for dose in map(float, args.doses):
            pert_gene, perturb_info = paired.perturb.gene_latent_add_shift(
                control=control,
                gene_idx=gene_idx,
                dose=float(dose),
                gat=gat,
                expr_stats=expr_stats,
                spatial_dim=spatial_dim,
                args=rt,
                device=device,
            )
            pert_state0 = control.control_state.detach().clone()
            pert_state0[:, spatial_dim:] = pert_gene
            _, pert_states = paired.record_rollout(
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
            pert_probs, pert_labels, _ = paired.perturb.classifier_readout(
                classifier,
                label_map,
                torch.as_tensor(pert_states[-1], dtype=torch.float32, device=device),
                terminal_input,
                spatial_dim,
                batch_size=int(args.classifier_batch_size),
            )
            counts, transition_long = transition_counts(base_labels, pert_labels)
            n_cells = int(len(base_labels))
            row = {
                "gene": gene,
                "dose": float(dose),
                "role": str(cand["role"]),
                "basis": str(cand["basis"]),
                "n_source_cells": n_cells,
                "any_terminal_label_change": int(np.sum(base_labels != pert_labels)),
                "any_terminal_label_change_rate": float(np.mean(base_labels != pert_labels)),
                "mean_delta_P_NeuB": float(np.mean(pert_probs[:, class_to_idx["NeuB"]] - base_probs[:, class_to_idx["NeuB"]])),
                "mean_delta_P_GlioB": float(np.mean(pert_probs[:, class_to_idx["GlioB"]] - base_probs[:, class_to_idx["GlioB"]])),
                **counts,
                **{f"{key}": float(value) for key, value in perturb_info.items()},
            }
            row["NeuB_to_GlioB_rate"] = float(row["NeuB_to_GlioB"] / n_cells)
            row["GlioB_to_NeuB_rate"] = float(row["GlioB_to_NeuB"] / n_cells)
            row["max_strict_b_to_b"] = int(max(row["NeuB_to_GlioB"], row["GlioB_to_NeuB"]))
            rows.append(row)
            for tr in transition_long:
                pair_rows.append(
                    {
                        "gene": gene,
                        "dose": float(dose),
                        **tr,
                    }
                )

    result = pd.DataFrame(rows)
    result = result.sort_values(["max_strict_b_to_b", "NeuB_to_GlioB", "GlioB_to_NeuB"], ascending=False).reset_index(drop=True)
    result_csv = out_dir / "literature_gene_b_to_b_transition_screen.csv"
    result.to_csv(result_csv, index=False)
    pair_csv = out_dir / "literature_gene_terminal_transition_pairs.csv"
    pd.DataFrame(pair_rows).to_csv(pair_csv, index=False)
    candidate_csv = out_dir / "literature_gene_candidate_availability.csv"
    candidates.to_csv(candidate_csv, index=False)

    figures = {
        "ranked_switches": plot_ranked_screen(result, candidates, out_dir, float(args.fig_width), float(args.fig_height)),
        "switch_heatmap": plot_heatmap(result, out_dir),
    }
    summary_out = {
        "status": "DONE",
        "definition": "Same E12.5 RGC source cells were rolled out under control and under each gene z+x perturbation; strict B-to-B terminal switches were counted by the trained spatiotemporal classifier.",
        "run_dir": str(run_dir),
        "trained_h5ad": str(summary["trained_h5ad"]),
        "classifier_path": str(classifier_path),
        "gat_checkpoint": str(gat.checkpoint_path),
        "source_time": float(args.source_time),
        "terminal_time": float(args.terminal_time),
        "n_source_cells": int(len(source_indices)),
        "screened_genes": screened_genes,
        "missing_or_unavailable_genes": candidates.loc[~candidates["screened_now"], "gene"].astype(str).tolist(),
        "doses": [float(v) for v in args.doses],
        "result_csv": str(result_csv),
        "transition_pair_csv": str(pair_csv),
        "candidate_csv": str(candidate_csv),
        "figures": figures,
    }
    summary_path = out_dir / "literature_gene_b_to_b_screen_summary.json"
    summary_path.write_text(json.dumps(summary_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_out


def main() -> None:
    args = parse_args()
    summary = make_screen(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

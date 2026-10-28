#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERTURB_SCRIPT_DIR = PROJECT_ROOT / "psg_full_ruot_gat" / "scripts"
if str(PERTURB_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PERTURB_SCRIPT_DIR))

import run_dorsal_gene_critical_perturbation as perturb  # noqa: E402


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "SpaPOT_runs"
    / "dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign"
)
DEFAULT_POOL = (
    PROJECT_ROOT
    / "result_figures"
    / "小鼠中脑"
    / "figure3_mdnb_module_member_pool_top50_v1"
    / "hvg2000_dnb_top50_module_member_pool.csv"
)
DEFAULT_GAT_CHECKPOINT = (
    DEFAULT_RUN_DIR
    / "gene_prior_gatae_retrain_20260523"
    / "gene_prior_gatae_10d.pt"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "result_figures"
    / "小鼠中脑"
    / "figure3_dnb_module_potential_descent_rank_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank DNB-module member genes by the local potential descent induced through "
            "the trained GAT encoder near the dorsal-midbrain critical interval."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--member-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--gat-checkpoint", type=Path, default=DEFAULT_GAT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source-times", nargs="+", type=float, default=[13.5, 13.7])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-source-cells", type=int, default=400)
    parser.add_argument("--seed", type=int, default=19491001)
    parser.add_argument("--gat-batch-size", type=int, default=16)
    parser.add_argument("--ode-chunk-size", type=int, default=512)
    parser.add_argument("--expression-clip-quantile", type=float, default=0.999)
    parser.add_argument("--finite-check-top", type=int, default=10)
    parser.add_argument("--finite-dose-sd", type=float, default=0.25)
    return parser.parse_args()


def helper_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        spatial_key="X_spatial_aligned",
        gene_input_key="X_gene_input",
        time_input_key="time_input",
        raw_time_key="time",
        annotation_key="Annotation",
        source_label="RGC",
        max_source_cells=int(args.max_source_cells),
        ode_chunk_size=int(args.ode_chunk_size),
        gat_batch_size=int(args.gat_batch_size),
    )


def potential_expression_gradient(
    *,
    model: torch.nn.Module,
    gat: Any,
    expression: torch.Tensor,
    spatial: torch.Tensor,
    time_input: float,
    expression_std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    std = torch.as_tensor(
        expression_std, dtype=expression.dtype, device=expression.device
    ).reshape(1, -1)
    for start in range(0, int(expression.shape[0]), int(batch_size)):
        expr = expression[start : start + int(batch_size)].detach().clone().requires_grad_(True)
        spatial_batch = spatial[start : start + int(batch_size)].detach()
        latent_raw = gat.model.encoder(expr)
        decoder_std = perturb.standardize_latent(gat, latent_raw)
        gene_state = perturb.decoder_to_dynamics_standardized(gat, decoder_std)
        state = torch.cat([spatial_batch, gene_state], dim=1)
        t = torch.tensor(float(time_input), dtype=state.dtype, device=state.device)
        potential = model.gene_velocity_net.potential(t, state).reshape(-1)
        gradient = torch.autograd.grad(potential.sum(), expr, create_graph=False)[0]
        rows.append((gradient * std).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def summarize_gradients(
    gradients: np.ndarray,
    gene_names: list[str],
) -> pd.DataFrame:
    median = np.median(gradients, axis=0)
    mean = np.mean(gradients, axis=0)
    q10 = np.quantile(gradients, 0.10, axis=0)
    q90 = np.quantile(gradients, 0.90, axis=0)
    preferred_sign = np.where(median <= 0.0, 1.0, -1.0)
    preferred_derivative = gradients * preferred_sign.reshape(1, -1)
    drop = -preferred_derivative
    absolute_gradient = np.abs(gradients)
    return pd.DataFrame(
        {
            "gene": gene_names,
            "potential_derivative_mean_per_sd": mean,
            "potential_derivative_median_per_sd": median,
            "potential_derivative_q10_per_sd": q10,
            "potential_derivative_q90_per_sd": q90,
            "preferred_expression_direction": np.where(
                preferred_sign > 0, "increase", "decrease"
            ),
            "preferred_direction_sign": preferred_sign.astype(int),
            "potential_drop_mean_per_sd": np.mean(drop, axis=0),
            "potential_drop_median_per_sd": np.median(drop, axis=0),
            "potential_drop_q10_per_sd": np.quantile(drop, 0.10, axis=0),
            "potential_drop_q90_per_sd": np.quantile(drop, 0.90, axis=0),
            "preferred_descent_fraction": np.mean(preferred_derivative < 0.0, axis=0),
            "mean_abs_potential_derivative_per_sd": np.mean(absolute_gradient, axis=0),
            "median_abs_potential_derivative_per_sd": np.median(
                absolute_gradient, axis=0
            ),
        }
    )


@torch.no_grad()
def exact_potential_delta(
    *,
    model: torch.nn.Module,
    gat: Any,
    expression: torch.Tensor,
    spatial: torch.Tensor,
    control_gene: torch.Tensor,
    time_input: float,
    gene_index: int,
    direction_sign: int,
    dose_sd: float,
    expression_std: np.ndarray,
    expression_lo: np.ndarray,
    expression_hi: np.ndarray,
    spatial_dim: int,
    batch_size: int,
) -> np.ndarray:
    control_state = torch.cat([spatial, control_gene], dim=1)
    control_u = perturb.potential_u(model, control_state, float(time_input))
    changed = expression.detach().clone()
    delta = float(direction_sign) * float(dose_sd) * float(expression_std[gene_index])
    lo = float(expression_lo[gene_index])
    hi = float(expression_hi[gene_index])
    changed[:, gene_index] = torch.clamp(changed[:, gene_index] + delta, min=lo, max=hi)
    perturbed_gene = perturb.encode_expression(
        gat,
        changed,
        batch_size=int(batch_size),
        device=expression.device,
    )
    perturbed_state = control_state.detach().clone()
    perturbed_state[:, spatial_dim:] = perturbed_gene
    perturbed_u = perturb.potential_u(model, perturbed_state, float(time_input))
    return (perturbed_u - control_u).detach().cpu().numpy().astype(np.float32)


def scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    perturb.seed_all(int(args.seed))
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    member_pool_path = args.member_pool.expanduser().resolve()
    member_pool = pd.read_csv(member_pool_path)
    member_genes = set(member_pool["gene"].astype(str))

    device = perturb.resolve_device(str(args.device))
    run_summary = perturb.load_json(run_dir / "summary.json")
    adata = ad.read_h5ad(run_summary["trained_h5ad"])
    helper = helper_args(args)
    expression = perturb.dense_float32(
        adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X
    )
    expression_stats = perturb.expression_stats(
        expression, float(args.expression_clip_quantile)
    )
    pairs = perturb.observed_time_pairs(adata, helper)
    initial_input = float(
        np.min(np.asarray(adata.obs[helper.time_input_key], dtype=np.float64))
    )
    spatial_dim = int(adata.obsm[helper.spatial_key].shape[1])
    method = str(run_summary.get("config", {}).get("ode_method", "rk4"))
    step_size = float(run_summary.get("config", {}).get("ode_step_size", 0.25))
    model = perturb.load_dynamics(run_summary, adata, helper, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gat_checkpoint = args.gat_checkpoint.expanduser().resolve()
    gat = perturb.load_gat_bundle(adata, device, gat_checkpoint)
    if list(map(str, adata.var_names)) != list(gat.gene_names):
        raise ValueError("HVG order does not match the GAT checkpoint.")

    source_indices = perturb.select_source_indices(
        adata, helper, initial_input, int(args.seed)
    )
    initial_state = perturb.build_initial_state(adata, helper, source_indices, device)
    initial_logw = torch.zeros(
        initial_state.shape[0], 1, dtype=torch.float32, device=device
    )
    lo = torch.as_tensor(
        expression_stats["lo"], dtype=torch.float32, device=device
    ).reshape(1, -1)
    hi = torch.as_tensor(
        expression_stats["hi"], dtype=torch.float32, device=device
    ).reshape(1, -1)

    gradients_by_time: list[np.ndarray] = []
    controls: dict[float, dict[str, Any]] = {}
    time_rows: list[pd.DataFrame] = []
    for raw_time in map(float, args.source_times):
        input_time = perturb.raw_to_input(raw_time, pairs)
        source_state, _ = perturb.integrate_model(
            model,
            initial_state,
            initial_logw,
            initial_input,
            input_time,
            method=method,
            step_size=step_size,
            chunk_size=int(args.ode_chunk_size),
            device=device,
        )
        decoded = perturb.decode_standardized_latent(
            gat,
            source_state[:, spatial_dim:],
            batch_size=int(args.gat_batch_size),
            device=device,
        )
        decoded = torch.minimum(torch.maximum(decoded, lo), hi)
        control_gene = perturb.encode_expression(
            gat,
            decoded,
            batch_size=int(args.gat_batch_size),
            device=device,
        )
        spatial = source_state[:, :spatial_dim].detach()
        gradient = potential_expression_gradient(
            model=model,
            gat=gat,
            expression=decoded,
            spatial=spatial,
            time_input=input_time,
            expression_std=expression_stats["std"],
            batch_size=int(args.gat_batch_size),
        )
        gradients_by_time.append(gradient)
        controls[raw_time] = {
            "input_time": float(input_time),
            "expression": decoded.detach(),
            "spatial": spatial,
            "control_gene": control_gene.detach(),
        }
        time_summary = summarize_gradients(gradient, list(gat.gene_names))
        time_summary.insert(1, "time", raw_time)
        time_summary.insert(2, "time_input", float(input_time))
        time_rows.append(time_summary)

    all_gradients = np.concatenate(gradients_by_time, axis=0)
    ranking = summarize_gradients(all_gradients, list(gat.gene_names))
    ranking["dnb_module_member_pool"] = ranking["gene"].isin(member_genes)
    ranking = ranking.merge(member_pool, on="gene", how="left", suffixes=("", "_pool"))
    ranking = ranking.sort_values(
        ["potential_drop_median_per_sd", "preferred_descent_fraction", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking.insert(1, "potential_descent_rank_all_hvg", np.arange(1, len(ranking) + 1))
    ranking["potential_descent_rank_dnb_pool"] = pd.Series(
        pd.NA, index=ranking.index, dtype="Int64"
    )
    pool_order = ranking[ranking["dnb_module_member_pool"]].index
    ranking.loc[pool_order, "potential_descent_rank_dnb_pool"] = np.arange(
        1, len(pool_order) + 1
    )

    dnb_ranking = ranking[ranking["dnb_module_member_pool"]].copy()
    full_output = out_dir / "hvg2000_potential_descent_ranking.csv"
    dnb_output = out_dir / "dnb299_potential_descent_ranking.csv"
    time_output = out_dir / "potential_descent_by_time.csv"
    finite_output = out_dir / "potential_descent_finite_difference_check.csv"
    ranking.to_csv(full_output, index=False)
    dnb_ranking.to_csv(dnb_output, index=False)
    pd.concat(time_rows, ignore_index=True).to_csv(time_output, index=False)

    check_genes = dnb_ranking.head(max(0, int(args.finite_check_top)))["gene"].tolist()
    for gene in ("Glis3", "Chd7", "Efna5", "Fzd3"):
        if gene in set(gat.gene_names) and gene not in check_genes:
            check_genes.append(gene)
    gene_index = {gene: index for index, gene in enumerate(gat.gene_names)}
    rank_lookup = ranking.set_index("gene")
    finite_rows: list[dict[str, Any]] = []
    for raw_time, control in controls.items():
        for gene in check_genes:
            row = rank_lookup.loc[gene]
            delta = exact_potential_delta(
                model=model,
                gat=gat,
                expression=control["expression"],
                spatial=control["spatial"],
                control_gene=control["control_gene"],
                time_input=float(control["input_time"]),
                gene_index=int(gene_index[gene]),
                direction_sign=int(row["preferred_direction_sign"]),
                dose_sd=float(args.finite_dose_sd),
                expression_std=expression_stats["std"],
                expression_lo=expression_stats["lo"],
                expression_hi=expression_stats["hi"],
                spatial_dim=spatial_dim,
                batch_size=int(args.gat_batch_size),
            )
            finite_rows.append(
                {
                    "gene": gene,
                    "time": float(raw_time),
                    "time_input": float(control["input_time"]),
                    "preferred_expression_direction": row[
                        "preferred_expression_direction"
                    ],
                    "finite_dose_sd": float(args.finite_dose_sd),
                    "delta_potential_mean": float(np.mean(delta)),
                    "delta_potential_median": float(np.median(delta)),
                    "delta_potential_q10": float(np.quantile(delta, 0.10)),
                    "delta_potential_q90": float(np.quantile(delta, 0.90)),
                    "fraction_potential_decreased": float(np.mean(delta < 0.0)),
                }
            )
    finite = pd.DataFrame(finite_rows)
    finite.to_csv(finite_output, index=False)

    focus: dict[str, dict[str, Any]] = {}
    for gene in ("Glis3", "Chd7", "Efna5", "Fzd3"):
        if gene not in rank_lookup.index:
            continue
        row = rank_lookup.loc[gene]
        check = finite[finite["gene"] == gene]
        focus[gene] = {
            "in_dnb_module_member_pool": bool(row["dnb_module_member_pool"]),
            "potential_descent_rank_all_hvg": scalar(
                row["potential_descent_rank_all_hvg"]
            ),
            "potential_descent_rank_dnb_pool": scalar(
                row["potential_descent_rank_dnb_pool"]
            ),
            "preferred_expression_direction": str(
                row["preferred_expression_direction"]
            ),
            "potential_drop_median_per_sd": scalar(
                row["potential_drop_median_per_sd"]
            ),
            "preferred_descent_fraction": scalar(row["preferred_descent_fraction"]),
            "finite_delta_potential_median_mean_across_times": (
                None
                if check.empty
                else float(check["delta_potential_median"].mean())
            ),
        }
    summary = {
        "status": "DONE",
        "interpretation": (
            "Local model-potential descent ranking within a DNB-module member pool; "
            "this is a dynamical prioritization score, not DNB evidence or causal validation."
        ),
        "definition": (
            "The gradient of U(s, encoder(expression), t) with respect to each gene is "
            "scaled by one observed-expression SD. The preferred sign is the expression "
            "increase or decrease giving negative local directional derivative."
        ),
        "source_times": list(map(float, args.source_times)),
        "source_population": "E12.5 RGC-origin trajectories",
        "n_source_cells": int(len(source_indices)),
        "hvg_count": int(len(gat.gene_names)),
        "dnb_module_member_pool_size": int(len(dnb_ranking)),
        "finite_difference_dose_sd": float(args.finite_dose_sd),
        "focus_genes": focus,
        "inputs": {
            "run_dir": str(run_dir),
            "trained_h5ad": str(run_summary["trained_h5ad"]),
            "model": str(run_summary["model_path"]),
            "gat_checkpoint": str(gat_checkpoint),
            "member_pool": str(member_pool_path),
        },
        "outputs": {
            "all_hvg_ranking": str(full_output),
            "dnb_pool_ranking": str(dnb_output),
            "by_time": str(time_output),
            "finite_difference_check": str(finite_output),
            "summary": str(out_dir / "potential_descent_summary.json"),
        },
    }
    summary_path = out_dir / "potential_descent_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    adata.file.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

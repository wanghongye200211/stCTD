#!/usr/bin/env python3
"""Prepare full-support spatial transport components without computing STE.

The seven existing main-figure cohorts and cached model states are preserved.
Dataset-specific cost scaling and scale-restoration conventions are explicit.
Axolotl compares current-cell-type measures at the two adjacent frames.
"""
from __future__ import annotations

import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = OUT / "runtime_cache"
for _key, _sub in {
    "KEOPS_CACHE_FOLDER": "keops", "PYKEOPS_CACHE_DIR": "keops",
    "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "matplotlib",
    "NUMBA_CACHE_DIR": "numba",
}.items():
    _path = CACHE / _sub
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_key] = str(_path)
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import hashlib
import argparse
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

MOUSE = ROOT / "SpaPOT_runs/dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign/stepu_spatialot_critical_dt0p1_current/trajectory_stepu_values.csv"
SIM = ROOT / "psg_full_ruot_gat/runs/pitchfork_fate_initbalanced_max48_hybrid_nocellprior_noalign_geneW10_e1000_s512_step0p1/stepu_spatialot_critical_geneW10_dt0p1/trajectory_stepu_values.csv"
FISH = ROOT / "result_figures/斑马鱼/canonical_transition_score_v1/rollout_values/zesta_single_origin_detail_mesoderm_values_dt0p2_model.csv"
AXO = ROOT / "result_figures/蝾螈/canonical_transition_score_v1/axolotl_centered_potential_trajectory_values.csv.gz"
AXO_OLD_CLASS = ROOT / "result_figures/axolotl_artista_preprocessed_injured_direct_ae_hybrid_hjb/e1000_s1024_lamhjb10_samplegrowth/stepu_spatialot_critical_dt0p5_ot2048/celltype_critical_score_dt0p5_score0p9_0p1_minframe20_stabilized/celltype_warning_summary.csv"

CONTRACTS = {
    "simulation_all": dict(cost_scale_rule="positive_pairwise_median", restore_scale=False, solver="sinkhorn", max_iter=500, stop_thr=1e-7, weight_floor=1e-12),
    "mouse_all": dict(cost_scale_rule="fixed", fixed_scale=2.4716388723941742, restore_scale=False, solver="sinkhorn", max_iter=500, stop_thr=1e-7, weight_floor=1e-12),
    "zebrafish_branch04": dict(cost_scale_rule="finite_pairwise_median_including_zero", restore_scale=True, solver="sinkhorn", max_iter=300, stop_thr=1e-6, weight_floor=0.0),
    "zebrafish_branch07": dict(cost_scale_rule="finite_pairwise_median_including_zero", restore_scale=True, solver="sinkhorn", max_iter=300, stop_thr=1e-6, weight_floor=0.0),
    "axolotl_reaEGC": dict(cost_scale_rule="positive_pairwise_median", restore_scale=False, solver="sinkhorn_stabilized", max_iter=1000, stop_thr=1e-7, weight_floor=1e-12),
    "axolotl_VLMC": dict(cost_scale_rule="positive_pairwise_median", restore_scale=False, solver="sinkhorn_stabilized", max_iter=1000, stop_thr=1e-7, weight_floor=1e-12),
    "axolotl_CMPN": dict(cost_scale_rule="positive_pairwise_median", restore_scale=False, solver="sinkhorn_stabilized", max_iter=1000, stop_thr=1e-7, weight_floor=1e-12),
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _load(path):
    data = pd.read_csv(path)
    required = {"frame", "time", "trajectory_id", "x", "y", "weight", "potential_U"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns in {path}: {required.difference(data.columns)}")
    if data.duplicated(["frame", "trajectory_id"]).any():
        raise ValueError(f"Duplicate frame/trajectory pairs in {path}")
    return data.sort_values(["frame", "trajectory_id"]).reset_index(drop=True)


def load_curves():
    """Return cached states for the seven existing figure cohorts.

    Values include data, label, time_unit, source_path, selection, full_frames.
    Full frames retains the original grid even if a current type is absent.
    """
    sim, mouse, fish, axo = map(_load, (SIM, MOUSE, FISH, AXO))
    curves = {}

    def add(name, data, full, label, unit, source, selection):
        curves[name] = {"data": data.copy(), "label": label, "time_unit": unit,
                        "source_path": str(source), "selection": selection,
                        "full_frames": full[["frame", "time"]].drop_duplicates().sort_values("frame").reset_index(drop=True)}

    add("simulation_all", sim, sim, "Simulation", "Model time", SIM, "all cached trajectories")
    add("mouse_all", mouse, mouse, "Mouse dorsal midbrain", "dpc", MOUSE, "all cached trajectories")
    source = fish[fish["source_group"].eq("Paraxial Mesoderm, Neural Keel")]
    final = source[source["frame"].eq(fish["frame"].max())]
    ids = final.loc[final["group"].isin(["Neural Crest", "Neural Crest, Otic Vesicle"]), "trajectory_id"]
    b04 = source[source["trajectory_id"].isin(ids)]
    b07 = fish[fish["source_group"].eq("Segmental Plate, Tail Bud")]
    add("zebrafish_branch04", b04, fish, "Zebrafish B04", "hpf", FISH,
        "source_group=Paraxial Mesoderm, Neural Keel; fixed ids with terminal group Neural Crest or Neural Crest, Otic Vesicle")
    add("zebrafish_branch07", b07, fish, "Zebrafish B07", "hpf", FISH,
        "source_group=Segmental Plate, Tail Bud; all source ids")
    for cell_type in ("reaEGC", "VLMC", "CMPN"):
        add("axolotl_" + cell_type, axo[axo["cell_type"].eq(cell_type)], axo,
            "Axolotl " + cell_type, "DPI", AXO, "current predicted cell_type=" + cell_type + "; each endpoint selected independently")
    return curves


def cost_scale(cost, contract):
    rule = contract["cost_scale_rule"]
    if rule == "fixed":
        return float(contract["fixed_scale"])
    if rule == "positive_pairwise_median":
        positive = cost[cost > 0]
        return max(float(np.median(positive)) if positive.size else 1.0, 1e-12)
    finite = cost[np.isfinite(cost)]
    scale = float(np.median(finite)) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(finite)) if finite.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return scale


def solve_interval(previous, current, contract, prior_result=None):
    """Return normalized transport-cost value and full convergence diagnostics."""
    import ot

    result = dict(spatial_raw=np.nan, cost_scale=np.nan, c_k=np.nan,
                  solver_iterations=0, solver_max_iterations=0, marginal_l1_error=np.nan,
                  row_marginal_l1_error=np.nan, column_marginal_l1_error=np.nan,
                  original_spatial_raw=np.nan, original_marginal_l1_error=np.nan,
                  solver_internal_error=np.nan, solver_method=contract["solver"], status="pending", attempts="[]")
    if previous.empty or current.empty:
        result["status"] = "empty_endpoint"
        return result
    required = ["x", "y", "weight"]
    if not (np.isfinite(previous[required].to_numpy(float)).all() and np.isfinite(current[required].to_numpy(float)).all()):
        result["status"] = "nonfinite_input"
        return result
    a0, b0 = previous["weight"].to_numpy(float), current["weight"].to_numpy(float)
    if np.maximum(a0, 0).sum() <= 0 or np.maximum(b0, 0).sum() <= 0:
        result["status"] = "zero_total_mass"
        return result
    a, b = np.maximum(a0, contract["weight_floor"]), np.maximum(b0, contract["weight_floor"])
    a, b = a / a.sum(), b / b.sum()
    cost = cdist(previous[["x", "y"]].to_numpy(float), current[["x", "y"]].to_numpy(float), metric="sqeuclidean")
    scale = cost_scale(cost, contract)
    cost /= scale
    c_k = scale if contract["restore_scale"] else 1.0
    result.update(cost_scale=scale, c_k=c_k)
    attempts = []
    if prior_result is not None:
        attempts = json.loads(prior_result["attempts"])
        for key in ("original_spatial_raw", "original_marginal_l1_error"):
            result[key] = prior_result[key]
    max_iterations = [contract["max_iter"], 2000, 10000, 50000]
    if prior_result is not None:
        max_iterations = []
    for limit in dict.fromkeys(max_iterations):
        # Keep the original solver/tolerance first. Increase only its iteration budget.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            plan, log = ot.sinkhorn(a, b, cost, reg=0.03, method=contract["solver"],
                                    numItermax=limit, stopThr=contract["stop_thr"],
                                    log=True, warn=True)
        row_error = float(np.abs(plan.sum(axis=1) - a).sum())
        column_error = float(np.abs(plan.sum(axis=0) - b).sum())
        error = max(row_error, column_error)
        internal_error = float(log["err"][-1]) if log.get("err") else np.nan
        value = float(np.sum(plan * cost) * c_k)
        niter = int(log.get("niter", log.get("n_iter", limit - 1))) + 1
        converged = bool(np.isfinite(plan).all() and np.isfinite(value) and
                         internal_error <= contract["stop_thr"] and
                         error <= contract["stop_thr"] * (np.sqrt(max(len(a), len(b))) + 1))
        attempt = dict(max_iterations=int(limit), iterations=niter, internal_error=internal_error,
                       marginal_l1_error=error, spatial_raw=value, converged=converged,
                       warnings=[str(item.message) for item in captured])
        attempts.append(attempt)
        if len(attempts) == 1:
            result.update(original_spatial_raw=value, original_marginal_l1_error=error)
        result.update(solver_iterations=niter, solver_max_iterations=int(limit),
                      marginal_l1_error=error, row_marginal_l1_error=row_error,
                      column_marginal_l1_error=column_error, solver_internal_error=internal_error)
        if converged:
            result.update(spatial_raw=value, status="ok_original_iterations" if len(attempts) == 1 else "ok_increased_iterations")
            break
    else:
        result["status"] = "not_converged"
    fixed_epsilon_exhausted = any(
        item.get("stage") == "fixed_epsilon_verification" and
        item.get("max_iterations", 0) >= 200000 for item in attempts)
    if result["status"] == "not_converged" and not fixed_epsilon_exhausted:
        # Epsilon scaling supplies a warm start only. The final plan is solved
        # and checked again at the original, fixed epsilon=0.03 and tolerance.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            warm_plan, warm_log = ot.bregman.sinkhorn_epsilon_scaling(
                a, b, cost, reg=0.03, epsilon0=1.0, numItermax=60,
                numInnerItermax=2000, stopThr=1e-12, log=True, warn=True)
        attempts.append(dict(stage="epsilon_scaling_warm_start", solver="sinkhorn_epsilon_scaling",
                             epsilon0=1.0, target_epsilon=0.03, outer_max_iterations=60,
                             outer_iterations=int(warm_log["niter"]) + 1, inner_max_iterations=2000,
                             stop_thr=1e-12, marginal_l1_error=float(max(np.abs(warm_plan.sum(1)-a).sum(), np.abs(warm_plan.sum(0)-b).sum())),
                             warnings=[str(item.message) for item in captured]))
        warmstart = warm_log["warmstart"]
        for limit in (10000, 50000, 200000):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                plan, log = ot.bregman.sinkhorn_stabilized(
                    a, b, cost, reg=0.03, warmstart=warmstart,
                    numItermax=limit, stopThr=contract["stop_thr"], log=True, warn=True)
            row_error = float(np.abs(plan.sum(axis=1) - a).sum())
            column_error = float(np.abs(plan.sum(axis=0) - b).sum())
            error = max(row_error, column_error)
            internal_error = float(log["err"][-1]) if log.get("err") else np.nan
            value = float(np.sum(plan * cost) * c_k)
            niter = int(log["n_iter"]) + 1
            converged = bool(np.isfinite(plan).all() and np.isfinite(value) and
                             internal_error <= contract["stop_thr"] and
                             error <= contract["stop_thr"] * (np.sqrt(max(len(a), len(b))) + 1))
            attempts.append(dict(stage="fixed_epsilon_verification", solver="sinkhorn_stabilized",
                                 epsilon=0.03, max_iterations=limit, iterations=niter,
                                 internal_error=internal_error, marginal_l1_error=error,
                                 spatial_raw=value, converged=converged,
                                 warnings=[str(item.message) for item in captured]))
            result.update(solver_iterations=niter, solver_max_iterations=limit,
                          solver_method="sinkhorn_stabilized_after_epsilon_scaling",
                          marginal_l1_error=error, row_marginal_l1_error=row_error,
                          column_marginal_l1_error=column_error, solver_internal_error=internal_error)
            if converged:
                result.update(spatial_raw=value, status="ok_solver_fallback")
                break
            warmstart = log["warmstart"]
    if result["status"] == "not_converged":
        # Solve exactly the same entropic objective in log-space. The LP is
        # only a dual initialization; the accepted plan is exponential at the
        # original epsilon and passes both marginals plus stationarity/KKT.
        from diagnostic_solver import solve_semidual

        plan, f, g, diagnostic = solve_semidual(
            a, b, cost, epsilon=0.03, target_l2=contract["stop_thr"])
        attempt = dict(stage="fixed_epsilon_entropic_semidual", solver="entropic_semidual",
                       **diagnostic)
        attempt["spatial_raw"] = diagnostic["normalized_transport_cost"] * c_k
        attempts.append(attempt)
        if diagnostic["verified"]:
            identity = hashlib.sha256(a.tobytes()+b.tobytes()+cost.tobytes()).hexdigest()[:16]
            plan_path = OUT / f"semidual_{identity}.npz"
            np.savez_compressed(plan_path, plan=plan, row_potential=f, column_potential=g,
                                log_plan=(f[:, None]+g[None, :]-cost)/0.03,
                                a=a, b=b, cost=cost, epsilon=0.03,
                                cost_scale=scale, c_k=c_k)
            result.update(
                spatial_raw=attempt["spatial_raw"], status="ok_entropic_semidual",
                solver_method="fixed_epsilon_entropic_semidual",
                solver_iterations=diagnostic["lbfgs_iterations"]+diagnostic["newton_iterations"],
                solver_max_iterations=20000,
                marginal_l1_error=max(diagnostic["row_l1"],diagnostic["column_l1"]),
                row_marginal_l1_error=diagnostic["row_l1"],
                column_marginal_l1_error=diagnostic["column_l1"],
                row_marginal_l2_error=diagnostic["row_l2"],
                column_marginal_l2_error=diagnostic["column_l2"],
                solver_internal_error=max(diagnostic["row_l2"],diagnostic["column_l2"]),
                stationarity_max_abs=diagnostic["stationarity_max_abs"],
                plan_log_roundtrip_kkt_max_abs=diagnostic["plan_log_roundtrip_kkt_max_abs"],
                signed_primal_dual_gap=diagnostic["signed_primal_dual_gap"],
                marginal_residual_gap_bound=diagnostic["marginal_residual_gap_bound"],
                semidual_plan_npz=plan_path.name, semidual_plan_sha256=sha256(plan_path))
    result["attempts"] = json.dumps(attempts, ensure_ascii=False)
    return result


def main():
    import ot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse already-converged intervals and retain previous failed attempts; source hashes must match.")
    args = parser.parse_args()

    curves = load_curves()
    source_manifest = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
                       for path in (SIM, MOUSE, FISH, AXO)}
    previous_rows = {}
    if args.resume:
        saved_manifest = json.loads((OUT / "spatial_manifest.json").read_text())
        if saved_manifest["contracts"] != CONTRACTS:
            raise ValueError("Cannot resume: spatial contracts changed")
        if sha256(OUT / "spatial_components.csv") != saved_manifest["outputs"]["spatial_components.csv"]["sha256"]:
            raise ValueError("Cannot resume: spatial output hash differs from manifest")
        for path, details in source_manifest.items():
            if details["sha256"] != saved_manifest["sources"][path]["sha256"]:
                raise ValueError(f"Cannot resume: source hash changed: {path}")
        previous_rows = {(row["curve_id"], int(row["frame"])): row for row in pd.read_csv(OUT / "spatial_components.csv").to_dict("records")}
    rows = []
    for curve_id, curve in curves.items():
        data = curve["data"]
        frames = curve["full_frames"].to_dict("records")
        for previous, current in zip(frames[:-1], frames[1:]):
            left = data[data["frame"].eq(previous["frame"])].sort_values("trajectory_id")
            right = data[data["frame"].eq(current["frame"])].sort_values("trajectory_id")
            prior = previous_rows.get((curve_id, int(current["frame"])))
            if prior is not None and str(prior["status"]).startswith("ok_"):
                result = {key: value for key, value in prior.items() if key not in ("curve_id", "frame", "time_prev", "time", "n_prev", "n_current")}
                result["solver_method"] = result.get("solver_method", CONTRACTS[curve_id]["solver"])
            else:
                result = solve_interval(left, right, CONTRACTS[curve_id], prior_result=prior)
            rows.append(dict(curve_id=curve_id, frame=int(current["frame"]),
                             time_prev=float(previous["time"]), time=float(current["time"]),
                             n_prev=len(left), n_current=len(right), **result))
            if len(rows) % 10 == 0:
                print(f"{curve_id}: frame {int(current['frame'])}, status={result['status']}, iterations={result['solver_iterations']}, marginal_l1={result['marginal_l1_error']:.3g}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "spatial_components.csv", index=False)
        print(f"Finished {curve_id}: {len(frames)-1} intervals", flush=True)
    table = pd.DataFrame(rows)
    # Record cached old class values for comparison without changing the output component.
    comparisons = []
    if AXO_OLD_CLASS.exists():
        old = pd.read_csv(AXO_OLD_CLASS)
        for cell_type in ("reaEGC", "VLMC", "CMPN"):
            check = table[table["curve_id"].eq("axolotl_" + cell_type)].merge(
                old.loc[old["cell_type"].eq(cell_type), ["frame", "celltype_spatial_ot_to_prev"]], on="frame")
            comparisons.append(dict(curve_id="axolotl_" + cell_type, n=len(check),
                                    max_absolute_difference=float(np.abs(check["spatial_raw"] - check["celltype_spatial_ot_to_prev"]).max())))
        source_manifest[str(AXO_OLD_CLASS)] = {"sha256": sha256(AXO_OLD_CLASS), "bytes": AXO_OLD_CLASS.stat().st_size, "role": "comparison_only"}
    plan_outputs = {}
    if "semidual_plan_npz" in table:
        for name in table["semidual_plan_npz"].dropna().unique():
            plan_outputs[str(name)] = {"sha256": sha256(OUT / str(name)), "role": "fixed-epsilon semidual stationary plan and dual certificate"}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha256(__file__),
                    solver_dependency_sha256={"diagnostic_solver.py": sha256(OUT / "diagnostic_solver.py")},
                    sources=source_manifest, contracts=CONTRACTS,
                    curves={key: {k: v for k, v in curve.items() if k not in ("data", "full_frames")} for key, curve in curves.items()},
                    sampling="full cached support; no subsampling; each current-cell-type endpoint selected independently",
                    epsilon=0.03, cost="squared Euclidean; no square root; no debiasing; no entropy term added to returned transport cost",
                    convergence="Original POT stopping tolerance retained; max(row L1,column L1) also must be <= stopThr*(sqrt(max(n,m))+1), consistent with the internal L2 tolerance. Original solver gets increased iteration budgets first. Unconverged intervals then use epsilon scaling as a warm start, followed by fixed-epsilon=0.03 stabilized Sinkhorn with the original tolerance. A final fixed-epsilon entropic semi-dual fallback uses LP dual initialization followed by L-BFGS and exact-Hessian Newton; both original marginals, stationarity, exp/log KKT and the primal-dual gap are checked. The LP plan is never used. Every attempt and warning is recorded in CSV.",
                    zero_mass_rule="If either original endpoint has no positive total mass, return NA. No uniform substitution.",
                    status_counts=table["status"].value_counts().to_dict(),
                    max_marginal_l1_error=float(table["marginal_l1_error"].max()),
                    celltype_cached_comparisons=comparisons,
                    packages=dict(numpy=np.__version__, pandas=pd.__version__, POT=ot.__version__),
                    outputs={"spatial_components.csv": {"sha256": sha256(OUT / "spatial_components.csv"), "rows": len(table)}, **plan_outputs},
                    scope="Spatial components only. No P aggregation, STE calculation, plotting, model retraining, or model hash claim.")
    (OUT / "spatial_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status_counts": manifest["status_counts"], "rows": len(table), "max_marginal_l1_error": manifest["max_marginal_l1_error"], "cached_comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

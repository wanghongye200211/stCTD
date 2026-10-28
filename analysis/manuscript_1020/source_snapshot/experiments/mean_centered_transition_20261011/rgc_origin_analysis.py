"""Fixed observed RGC-origin potential and matched spatial readouts."""
from pathlib import Path
import os
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mouse_rgc_origin_20261011"
OUT.mkdir(parents=True, exist_ok=True)
POTENTIAL_WEIGHT = 0.99
SPATIAL_WEIGHT = 0.01
os.environ["MPLCONFIGDIR"] = str(OUT / "runtime_cache/matplotlib")
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(ROOT / "outputs/critical_curves_redefined_20260915_w08_w02"))
import prepare_spatial as spatial
spatial.OUT = OUT / "audit/solver"
spatial.OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute():
    run = ROOT / "SpaPOT_runs/dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign"
    trajectory = run / "stepu_spatialot_critical_dt0p1_current/trajectory_stepu_values.csv"
    members_path = ROOT / "outputs/centering_comparison_20261007/source_data/mouse_initial_membership.csv"
    previous_audit = json.loads((ROOT / "outputs/centering_comparison_20261007/qa/mouse_rgc_origin_audit.json").read_text())
    # Retain the verified observed-label to trajectory-ID mapping.
    for p in [trajectory, run / "stvcr_notebook_trained.h5ad"]:
        assert sha(p) == previous_audit["source_sha256"][str(p.relative_to(ROOT))]
    members = pd.read_csv(members_path)
    ids = members.loc[members.initial_annotation.eq("RGC"), "trajectory_id"].to_numpy()
    assert len(ids) == len(set(ids)) == 1708
    data = pd.read_csv(trajectory).sort_values(["frame", "trajectory_id"])
    assert not data.duplicated(["frame", "trajectory_id"]).any()
    cohort = data[data.trajectory_id.isin(ids)].copy()
    assert (cohort.groupby("frame").size() == len(ids)).all()
    assert len(cohort.frame.unique()) == 41
    for sub in ["source_data", "figures", "audit"]:
        (OUT / sub).mkdir(exist_ok=True)
    members[members.trajectory_id.isin(ids)].to_csv(OUT / "source_data/initial_rgc_membership.csv", index=False)
    cohort.to_csv(OUT / "source_data/rgc_origin_trajectories.csv.gz", index=False, compression="gzip")
    contract = {
        "scope": "User-selected fixed-RGC-origin readout with spatial weight 0.01; selection applies to this analysis, not a cross-dataset change.",
        "status": "weights_selected_by_user_2026-10-17",
        "selected_potential_method": "raw",
        "cohort": "Observed E12.5 RGC annotation, 1708 fixed trajectory IDs through E16.5, including all predicted descendants regardless of later type.",
        "reference_population": "Full saved mouse population of 2237 particles for mean centering; no RGC-specific recentering.",
        "potential": "Absolute same-ID adjacent increments; unweighted q90; raw and full-population-mean-centered variants.",
        "retrospective": "Centered nine-node mean of potential only, min_periods=1.",
        "past_processing": "Trailing nine-node mean of potential only, min_periods=1; saved model still fitted all observation times.",
        "spatial": "Recomputed on the same 1708 fixed IDs at both endpoints; growth weights normalized within this cohort at each endpoint; squared Euclidean cost; same fixed mouse cost scale; epsilon 0.03; linear cost of entropic plan, without square root or debiasing.",
        "spatial_contract": dict(spatial.CONTRACTS["mouse_all"]),
        "normalization": "Each displayed component divided by its own first displayed interval value. No maximum normalization.",
        "joint_score": "0.99 * potential / first + 0.01 * spatial / first; same cohort in both components.",
        "weights": {"potential": POTENTIAL_WEIGHT, "spatial": SPATIAL_WEIGHT},
        "inference": "Matched score diagnostics are exported separately in outputs/mouse_rgc_mainstyle_20261017. They describe the saved model curve, not a biological critical-point probability.",
        "figure_contract": {
            "backend": "Established Python/matplotlib workflow",
            "claim": "Show direct potential change and spatial redistribution for a fixed observed RGC-origin cohort, and isolate the effect of mean centering.",
            "archetype": "quantitative grid", "width_mm": 183,
            "panels": "Main: direct potential, matched spatial cost, direct joint score. Sensitivity: potential and joint score, raw versus mean-centered.",
            "exports": ["PNG", "PDF", "SVG"], "editable_text": True,
            "risks": "Retrospective smoothing; model-derived trajectories; cohort selected during exploration; varying y-ranges can exaggerate small changes.",
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [trajectory, members_path]},
    }
    contract_path = OUT / "contract.json"
    if (OUT / "source_data/spatial_intervals.csv").exists():
        old = json.loads(contract_path.read_text())
        for field in ["cohort", "spatial_contract", "source_sha256"]:
            assert old[field] == contract[field], "Cached spatial inputs changed: " + field
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    potential = data.pivot(index="frame", columns="trajectory_id", values="potential_U").sort_index()
    raw = potential[ids].diff().iloc[1:]
    centered = potential[ids].sub(potential.mean(axis=1), axis=0).diff().iloc[1:]
    frames = potential.index.to_numpy()[1:]
    times = data.groupby("frame").time.first().loc[frames].to_numpy()
    scores = pd.DataFrame({"frame": frames, "time": times,
                           "raw_q90": np.quantile(np.abs(raw.to_numpy()), .9, axis=1),
                           "centered_q90": np.quantile(np.abs(centered.to_numpy()), .9, axis=1)})
    sc_path = OUT / "source_data/spatial_intervals.csv"
    cached = pd.read_csv(sc_path) if sc_path.exists() else pd.DataFrame()
    rows = []
    with threadpool_limits(limits=1):
        for f in frames:
            if len(cached) and f in set(cached.frame):
                row = cached[cached.frame.eq(f)].iloc[0].to_dict()
                assert str(row["status"]).startswith("ok")
            else:
                prev = cohort[cohort.frame.eq(f-1)].sort_values("trajectory_id")
                curr = cohort[cohort.frame.eq(f)].sort_values("trajectory_id")
                assert np.array_equal(prev.trajectory_id, curr.trajectory_id)
                row = spatial.solve_interval(prev, curr, contract["spatial_contract"])
                if not row["status"].startswith("ok"):
                    raise RuntimeError(f"Unverified spatial interval {f}: {row['status']}")
                row.update(frame=int(f), time=float(curr.time.iloc[0]), n_previous=len(prev), n_current=len(curr))
            rows.append(row)
            pd.DataFrame(rows).to_csv(sc_path, index=False)
            if f % 5 == 0 or f == 1:
                print(f"Spatial interval {f}/40 verified", flush=True)
    sc = pd.DataFrame(rows).set_index("frame").loc[frames]
    scores["spatial_raw"] = sc.spatial_raw.to_numpy()
    scores["spatial_relative_first"] = scores.spatial_raw / scores.spatial_raw.iloc[0]
    summary = []
    for mode, center in [("retrospective", True), ("past", False)]:
        for method in ["raw", "centered"]:
            smooth = scores[method + "_q90"].rolling(9, center=center, min_periods=1).mean()
            assert smooth.iloc[0] > 0
            key = f"{method}_{mode}"
            scores[key + "_potential"] = smooth / smooth.iloc[0]
            scores[key + "_joint"] = POTENTIAL_WEIGHT * scores[key + "_potential"] + SPATIAL_WEIGHT * scores.spatial_relative_first
            for component in ["potential", "joint"]:
                y = scores[key + "_" + component]
                summary.append({"method": method, "processing": mode, "component": component,
                                "peak_time": float(scores.time.iloc[y.argmax()]), "peak_relative_first": float(y.max()),
                                "last_relative_first": float(y.iloc[-1])})
    scores.to_csv(OUT / "source_data/rgc_origin_scores.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "source_data/summary.csv", index=False)
    reference = pd.read_csv(ROOT / "outputs/mouse_raw_difference_audit_20261011/fixed_origin_cohort_curves.csv")
    replay = []
    for method, label in [("raw", "raw"), ("centered", "mean_centered_full_population")]:
        ref = reference[reference.cohort.eq("RGC") & reference.method.eq(label)].sort_values("time")
        err = float(np.max(np.abs(ref.q90_smooth9_relative_first.to_numpy() - scores[method + "_retrospective_potential"])))
        assert err < 1e-10
        replay.append({"method": method, "max_error": err})
    assert np.isfinite(scores.to_numpy()).all()
    assert np.allclose(scores.filter(regex="potential$|joint$|relative_first$").iloc[0], 1)
    audit = {"fixed_ids": len(ids), "frames": 41, "verified_spatial_intervals": len(sc),
             "max_marginal_l1_error": float(sc.marginal_l1_error.max()),
             "potential_replay": replay, "first_values_equal_one": True}
    (OUT / "audit/numeric_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    compute()

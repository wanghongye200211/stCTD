#!/usr/bin/env python3
"""Redraw the selected manuscript curves after explicit aggregation selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl_cache"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from prepare_spatial import load_curves, CONTRACTS
sys.path.insert(0, str(ROOT / "scripts"))
from critical_transition_contract import local_window_welch

WEIGHTS = (0.8, 0.2)
COLORS = {"STE": "#253A52", "potential": "#B97557", "spatial": "#5C97AB"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate(aggregation):
    audit = json.loads((HERE / "potential_cache_audit.json").read_text())
    spatial_manifest = json.loads((HERE / "spatial_manifest.json").read_text())
    if sha256(HERE / "spatial_components.csv") != spatial_manifest["outputs"]["spatial_components.csv"]["sha256"]:
        raise ValueError("Spatial component file differs from its completed preparation manifest")
    for name, record in audit.items():
        if sha256(ROOT / record["source"]) != record["sha256"]:
            raise ValueError(f"{name}: raw source changed after endpoint audit")
        spatial_hash = spatial_manifest["sources"][str(ROOT / record["source"])]["sha256"]
        if spatial_hash != record["sha256"]:
            raise ValueError(f"{name}: spatial and potential use different source snapshots")
    spatial = pd.read_csv(HERE / "spatial_components.csv")
    curves = load_curves()
    scored = []
    metadata = {}
    for curve_id, info in curves.items():
        values = info["data"].copy()
        frames = np.sort(values.frame.unique())
        if frames[0] != 0:
            raise ValueError(f"{curve_id}: source frame zero missing")
        # Cached deltas were calculated before current-type selection. Recomputing
        # diff after selection would incorrectly jump across absent memberships.
        values = values.loc[values.frame > 0]
        if values.duplicated(["frame", "trajectory_id"]).any():
            raise ValueError(f"{curve_id}: duplicate particle/frame")
        if not np.isfinite(values.abs_step_delta_U).all():
            raise ValueError(f"{curve_id}: nonfinite raw potential change")
        grouped = values.groupby(["frame", "time"])["abs_step_delta_U"]
        if aggregation != "mean":
            raise ValueError("The confirmed potential aggregation is the unweighted arithmetic mean")
        p = grouped.mean()
        p = p.rename("potential_raw").reset_index()
        p["potential_n"] = grouped.size().to_numpy()
        s = spatial.loc[spatial.curve_id == curve_id].copy()
        expected_frames = np.arange(1, int(frames[-1]) + 1)
        if not np.array_equal(np.sort(s.frame.to_numpy()), expected_frames):
            raise ValueError(f"{curve_id}: spatial frame grid is incomplete or duplicated")
        if not np.array_equal(np.sort(p.frame.to_numpy()), expected_frames):
            raise ValueError(f"{curve_id}: potential frame grid is incomplete or duplicated")
        part = s.merge(p, on="frame", how="left", suffixes=("", "_potential"), validate="one_to_one")
        if not np.allclose(part.time, part.time_potential, atol=1e-8, equal_nan=False):
            raise ValueError(f"{curve_id}: time mismatch")
        part = part.drop(columns="time_potential").sort_values("frame")
        for name in ("potential_raw", "spatial_raw"):
            if not np.isfinite(part[name]).all() or (part[name] < 0).any():
                raise ValueError(f"{curve_id}: {name} contains invalid values")
        allowed_l1 = CONTRACTS[curve_id]["stop_thr"] * (np.sqrt(np.maximum(part.n_prev, part.n_current)) + 1)
        if not np.isfinite(part.marginal_l1_error).all() or (part.marginal_l1_error > allowed_l1).any() or not part.status.str.startswith("ok_").all():
            raise ValueError(f"{curve_id}: transport marginals have not converged")
        baseline = part.loc[part.frame == 1]
        if len(baseline) != 1:
            raise ValueError(f"{curve_id}: first interval missing")
        p0, s0 = float(baseline.potential_raw.iloc[0]), float(baseline.spatial_raw.iloc[0])
        if not (np.isfinite(p0) and np.isfinite(s0) and p0 > 0 and s0 > 0):
            raise ValueError(f"{curve_id}: undefined first-interval normalization; P={p0}, S={s0}")
        part["potential_reference"] = p0
        part["spatial_reference"] = s0
        part["potential_normalized"] = part.potential_raw / p0
        part["spatial_normalized"] = part.spatial_raw / s0
        part["potential_contribution"] = WEIGHTS[0] * part.potential_normalized
        part["spatial_contribution"] = WEIGHTS[1] * part.spatial_normalized
        part["STE"] = part.potential_contribution + part.spatial_contribution
        if not np.isfinite(part.STE).all() or (part.STE < 0).any():
            raise ValueError(f"{curve_id}: invalid STE")
        if not np.isclose(part.loc[part.frame == 1, "STE"].iloc[0], 1.0):
            raise ValueError("First interval STE must equal one")
        part["aggregation"] = aggregation
        tests = local_window_welch(part.time, part.STE, window_points=4)
        part = part.reset_index(drop=True)
        for name in tests.columns:
            if name != "time":
                part[name] = tests[name].to_numpy()
        metadata[curve_id] = {"label": info["label"], "time_unit": info["time_unit"],
                              "population_selection": info["selection"],
                              "P_reference": p0, "S_reference": s0,
                              "rows": len(part), "peak_time": float(part.loc[part.STE.idxmax(), "time"]),
                              "peak_STE": float(part.STE.max())}
        scored.append(part)
    return pd.concat(scored, ignore_index=True), metadata


def style_axis(ax, time_unit):
    ax.set_xlabel("Model time" if time_unit == "Model time" else f"Time ({time_unit})")
    ax.set_ylabel("STE")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(length=2.5, width=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1, color="#B5B5B5", linestyle=(0, (3, 3)), linewidth=0.65, zorder=0)
    ax.margins(x=0.025)


def save(fig, base):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(base.with_suffix("." + suffix), dpi=450, facecolor="white")
    plt.close(fig)


def plot(data, metadata, out):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                        "axes.linewidth": 0.65, "svg.fonttype": "none", "pdf.fonttype": 42,
                        "legend.frameon": False, "lines.linewidth": 1.6})
    overview, axes = plt.subplots(4, 2, figsize=(7.2, 8.2), layout="constrained")
    for index, (curve_id, meta) in enumerate(metadata.items()):
        sub = data.loc[data.curve_id == curve_id]
        ax = axes.flat[index]
        ax.plot(sub.time, sub.STE, color=COLORS["STE"])
        ax.set_title(meta["label"], loc="left", fontweight="bold")
        style_axis(ax, meta["time_unit"])
        ax.set_ylim(bottom=0)
        single, sax = plt.subplots(figsize=(3.35, 2.65), layout="constrained")
        sax.plot(sub.time, sub.STE, color=COLORS["STE"])
        sax.set_title(meta["label"], loc="left", fontweight="bold")
        style_axis(sax, meta["time_unit"])
        sax.set_ylim(bottom=0)
        save(single, out / curve_id)
        components, cax = plt.subplots(figsize=(3.6, 2.9), layout="constrained")
        cax.plot(sub.time, sub.STE, color=COLORS["STE"], label="STE")
        cax.plot(sub.time, sub.potential_contribution, color=COLORS["potential"], label="Potential contribution (0.8)")
        cax.plot(sub.time, sub.spatial_contribution, color=COLORS["spatial"], label="Spatial contribution (0.2)")
        style_axis(cax, meta["time_unit"])
        cax.set_ylabel("Score / contribution")
        cax.set_title(meta["label"], loc="left", fontweight="bold")
        cax.legend(loc="best", fontsize=6.7, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
        cax.set_ylim(bottom=0)
        save(components, out / (curve_id + "_components"))
    axes.flat[7].axis("off")
    axes.flat[7].text(0, .85, "Potential: class/cohort mean\nWeights: potential 0.8 / spatial 0.2\nReference: first interval\nDashed line: STE = 1", transform=axes.flat[7].transAxes,
                      va="top", fontsize=9, linespacing=1.7)
    save(overview, out / "critical_curves_overview")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation", choices=["mean"], default="mean",
                        help="User-confirmed unweighted arithmetic mean of per-particle absolute net changes.")
    args = parser.parse_args()
    out = HERE / args.aggregation
    out.mkdir(exist_ok=True)
    data, metadata = calculate(args.aggregation)
    data.to_csv(out / "curve_source_data.csv", index=False)
    plot(data, metadata, out)
    manifest = {"status": "generated_requires_visual_QA", "aggregation": args.aggregation,
                "potential": "User-confirmed unweighted arithmetic mean of per-particle absolute raw endpoint potential differences; no frame centering",
                "weights": {"potential": .8, "spatial": .2}, "normalization": "each component divided by its own first reconstructed interval",
                "smooth_clip_interpolate": False, "curve_metadata": metadata,
                "statistics": "Exploratory four-point recent vs preceding four-point one-sided Welch test, BH within each curve; reconstructed time points are autocorrelated, not biological replicates. No significance markers in plots.",
                "input_sha256": {"spatial_components.csv": sha256(HERE / "spatial_components.csv"),
                                  "spatial_manifest.json": sha256(HERE / "spatial_manifest.json"),
                                  "potential_cache_audit.json": sha256(HERE / "potential_cache_audit.json")},
                "raw_sources": json.loads((HERE / "potential_cache_audit.json").read_text()),
                "script_sha256": {p.name: sha256(p) for p in [Path(__file__), HERE / "prepare_spatial.py", ROOT / "scripts/critical_transition_contract.py"]},
                "outputs_sha256": {p.name: sha256(p) for p in out.iterdir() if p.suffix in {".png", ".pdf", ".svg", ".csv"}}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

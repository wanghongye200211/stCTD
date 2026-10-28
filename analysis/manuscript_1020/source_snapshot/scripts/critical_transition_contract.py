#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_POTENTIAL_WEIGHT = 0.9
DEFAULT_SPATIAL_WEIGHT = 0.1
DEFAULT_WINDOW_POINTS = 4
DEFAULT_LOWER_PERCENTILE = 5.0
DEFAULT_UPPER_PERCENTILE = 95.0


def robust_scale_01(
    values: pd.Series | np.ndarray,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE,
) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return result, {"lower": np.nan, "upper": np.nan}
    lower, upper = np.nanpercentile(
        array[finite], [float(lower_percentile), float(upper_percentile)]
    )
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        result[finite] = 0.0
        return result, {"lower": float(lower), "upper": float(upper)}
    clipped = np.clip(array[finite], lower, upper)
    result[finite] = (clipped - lower) / (upper - lower)
    return result, {"lower": float(lower), "upper": float(upper)}


def centered_stepu_q90(
    particles: pd.DataFrame,
    *,
    time_col: str = "time",
    trajectory_col: str = "trajectory_id",
    potential_col: str = "potential_U",
    quantile: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {time_col, trajectory_col, potential_col}
    missing = required.difference(particles.columns)
    if missing:
        raise KeyError(f"Particle table is missing columns: {sorted(missing)}")
    work = particles.copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work[potential_col] = pd.to_numeric(work[potential_col], errors="coerce")
    work = work.dropna(subset=[time_col, trajectory_col, potential_col]).sort_values(
        [trajectory_col, time_col]
    )
    duplicate = work.duplicated([trajectory_col, time_col], keep=False)
    if duplicate.any():
        raise ValueError("StepU requires one potential value per trajectory and time point.")

    frame_median = work.groupby(time_col)[potential_col].transform("median")
    work["potential_centered"] = work[potential_col] - frame_median
    work["delta_potential_centered"] = work.groupby(trajectory_col)[
        "potential_centered"
    ].diff()
    work["abs_delta_potential_centered"] = work["delta_potential_centered"].abs()
    summary = (
        work.dropna(subset=["abs_delta_potential_centered"])
        .groupby(time_col, as_index=False)
        .agg(
            potential_change_raw=(
                "abs_delta_potential_centered",
                lambda values: float(np.quantile(values, float(quantile))),
            ),
            matched_trajectory_count=("abs_delta_potential_centered", "size"),
            potential_frame_median=(potential_col, "median"),
        )
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    return summary, work


def benjamini_hochberg(pvalues: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted
    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    m = ranked.size
    q_ranked = ranked * m / np.arange(1, m + 1, dtype=float)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_finite = np.empty_like(q_ranked)
    q_finite[order] = q_ranked
    adjusted[finite_indices] = q_finite
    return adjusted


def local_window_welch(
    times: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    *,
    window_points: int = DEFAULT_WINDOW_POINTS,
) -> pd.DataFrame:
    time_values = np.asarray(times, dtype=float)
    score_values = np.asarray(scores, dtype=float)
    if time_values.shape != score_values.shape:
        raise ValueError("Times and scores must have the same shape.")
    n = int(window_points)
    if n < 2:
        raise ValueError("window_points must be at least 2 for a Welch test.")

    rows: list[dict[str, Any]] = []
    for end in range(score_values.size):
        row: dict[str, Any] = {
            "time": float(time_values[end]),
            "window_points": n,
            "earlier_start_index": np.nan,
            "earlier_end_index": np.nan,
            "recent_start_index": np.nan,
            "recent_end_index": np.nan,
            "earlier_mean": np.nan,
            "recent_mean": np.nan,
            "mean_shift": np.nan,
            "welch_t_greater": np.nan,
            "p_value_greater": np.nan,
        }
        if end + 1 >= 2 * n:
            earlier_start = end + 1 - 2 * n
            earlier_end = end + 1 - n
            recent_start = earlier_end
            recent_end = end + 1
            earlier = score_values[earlier_start:earlier_end]
            recent = score_values[recent_start:recent_end]
            if np.isfinite(earlier).all() and np.isfinite(recent).all():
                test = stats.ttest_ind(
                    recent,
                    earlier,
                    equal_var=False,
                    alternative="greater",
                )
                row.update(
                    {
                        "earlier_start_index": earlier_start,
                        "earlier_end_index": earlier_end - 1,
                        "recent_start_index": recent_start,
                        "recent_end_index": recent_end - 1,
                        "earlier_mean": float(np.mean(earlier)),
                        "recent_mean": float(np.mean(recent)),
                        "mean_shift": float(np.mean(recent) - np.mean(earlier)),
                        "welch_t_greater": float(test.statistic),
                        "p_value_greater": float(test.pvalue),
                    }
                )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(result["p_value_greater"])
    result["positive_shift"] = result["mean_shift"] > 0.0
    return result


def score_one_group(
    table: pd.DataFrame,
    *,
    time_col: str,
    potential_col: str,
    spatial_col: str,
    potential_weight: float,
    spatial_weight: float,
    window_points: int,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = table.copy().sort_values(time_col).reset_index(drop=True)
    for column in (time_col, potential_col, spatial_col):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[time_col, potential_col, spatial_col]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No finite component rows remain after numeric conversion.")

    potential_scaled, potential_limits = robust_scale_01(
        work[potential_col], lower_percentile, upper_percentile
    )
    spatial_scaled, spatial_limits = robust_scale_01(
        work[spatial_col], lower_percentile, upper_percentile
    )
    work["potential_change_scaled"] = potential_scaled
    work["spatial_ot_scaled"] = spatial_scaled
    work["transition_score"] = (
        float(potential_weight) * work["potential_change_scaled"]
        + float(spatial_weight) * work["spatial_ot_scaled"]
    )
    tests = local_window_welch(
        work[time_col], work["transition_score"], window_points=int(window_points)
    )
    test_columns = [column for column in tests.columns if column != "time"]
    work = pd.concat([work, tests[test_columns]], axis=1)
    metadata = {
        "row_count": int(len(work)),
        "potential_limits": potential_limits,
        "spatial_limits": spatial_limits,
        "finite_test_count": int(work["p_value_greater"].notna().sum()),
        "minimum_raw_p": (
            None
            if not work["p_value_greater"].notna().any()
            else float(work["p_value_greater"].min())
        ),
        "minimum_bh_q": (
            None if not work["q_value_bh"].notna().any() else float(work["q_value_bh"].min())
        ),
    }
    return work, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the locked transition-score and local-window p-value contract to a component CSV."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--potential-col", required=True)
    parser.add_argument("--spatial-col", required=True)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--drop-first-per-group", action="store_true")
    parser.add_argument("--potential-weight", type=float, default=DEFAULT_POTENTIAL_WEIGHT)
    parser.add_argument("--spatial-weight", type=float, default=DEFAULT_SPATIAL_WEIGHT)
    parser.add_argument("--window-points", type=int, default=DEFAULT_WINDOW_POINTS)
    parser.add_argument("--lower-percentile", type=float, default=DEFAULT_LOWER_PERCENTILE)
    parser.add_argument("--upper-percentile", type=float, default=DEFAULT_UPPER_PERCENTILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isclose(float(args.potential_weight) + float(args.spatial_weight), 1.0):
        raise ValueError("Potential and spatial weights must sum to one.")
    source = pd.read_csv(args.input_csv.expanduser().resolve())
    required = {args.time_col, args.potential_col, args.spatial_col}
    if args.group_col:
        required.add(args.group_col)
    missing = required.difference(source.columns)
    if missing:
        raise KeyError(f"Input CSV is missing columns: {sorted(missing)}")

    groups: list[tuple[str, pd.DataFrame]]
    if args.group_col:
        groups = [
            (str(name), group.copy())
            for name, group in source.groupby(args.group_col, sort=False, dropna=False)
        ]
    else:
        groups = [("all", source.copy())]

    outputs: list[pd.DataFrame] = []
    group_metadata: dict[str, Any] = {}
    for name, group in groups:
        group = group.sort_values(args.time_col).reset_index(drop=True)
        if args.drop_first_per_group and not group.empty:
            group = group.iloc[1:].reset_index(drop=True)
        scored, metadata = score_one_group(
            group,
            time_col=str(args.time_col),
            potential_col=str(args.potential_col),
            spatial_col=str(args.spatial_col),
            potential_weight=float(args.potential_weight),
            spatial_weight=float(args.spatial_weight),
            window_points=int(args.window_points),
            lower_percentile=float(args.lower_percentile),
            upper_percentile=float(args.upper_percentile),
        )
        if args.group_col:
            scored[str(args.group_col)] = name
        outputs.append(scored)
        group_metadata[name] = metadata

    output = pd.concat(outputs, ignore_index=True)
    sort_columns = [str(args.group_col), str(args.time_col)] if args.group_col else [str(args.time_col)]
    output = output.sort_values(sort_columns).reset_index(drop=True)
    output_path = args.output_csv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    manifest_path = (
        args.manifest_json.expanduser().resolve()
        if args.manifest_json is not None
        else output_path.with_suffix(".manifest.json")
    )
    manifest = {
        "status": "DONE",
        "input_csv": str(args.input_csv.expanduser().resolve()),
        "output_csv": str(output_path),
        "definition": {
            "potential_component": (
                "q90 absolute adjacent change of frame-median-centred potential on matched trajectories"
            ),
            "spatial_component": (
                "balanced spatial Sinkhorn OT between adjacent frame measures with within-frame normalised mass"
            ),
            "component_scaling": (
                f"whole-trajectory clipped {float(args.lower_percentile):g}-{float(args.upper_percentile):g} "
                "percentile min-max to [0,1]"
            ),
            "score": (
                f"{float(args.potential_weight):g} * potential_change_scaled + "
                f"{float(args.spatial_weight):g} * spatial_ot_scaled"
            ),
            "post_sum_normalization": False,
            "p_value": (
                f"one-sided Welch test, recent {int(args.window_points)} score points greater than "
                f"the immediately preceding {int(args.window_points)} points"
            ),
            "multiplicity": "Benjamini-Hochberg across all finite local-window tests within each group",
        },
        "columns": {
            "time": str(args.time_col),
            "potential_raw": str(args.potential_col),
            "spatial_raw": str(args.spatial_col),
            "group": None if args.group_col is None else str(args.group_col),
        },
        "drop_first_per_group": bool(args.drop_first_per_group),
        "group_metadata": group_metadata,
        "interpretation": (
            "P values quantify local shifts among autocorrelated reconstructed-curve points; "
            "they are not biological-replicate significance tests."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

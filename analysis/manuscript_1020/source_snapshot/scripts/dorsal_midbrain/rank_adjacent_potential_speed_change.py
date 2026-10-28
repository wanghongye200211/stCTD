#!/usr/bin/env python3
"""Rank HVGs by absolute potential-descent speed change between two times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOCUS_GENES = ("Glis3", "Chd7", "Fzd3", "Efna5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--finite-check", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-time", type=float, required=True)
    parser.add_argument("--previous-time", type=float, required=True)
    return parser.parse_args()


def time_slice(source: pd.DataFrame, time: float) -> pd.DataFrame:
    result = source.loc[np.isclose(source["time"], time)].copy()
    if result.empty:
        raise ValueError(f"No rows found at time {time:g}")
    if result["gene"].duplicated().any():
        raise ValueError(f"Duplicate genes found at time {time:g}")
    return result.set_index("gene", verify_integrity=True)


def time_tag(time: float) -> str:
    return f"e{time:.1f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    decision = time_slice(source, float(args.decision_time))
    previous = time_slice(source, float(args.previous_time))
    if set(decision.index) != set(previous.index):
        raise ValueError("Decision and previous times do not contain the same genes.")
    previous = previous.loc[decision.index]

    result = pd.DataFrame(index=decision.index)
    result.index.name = "gene"
    result["decision_time"] = float(args.decision_time)
    result["previous_time"] = float(args.previous_time)
    result["preferred_expression_direction_at_decision"] = decision[
        "preferred_expression_direction"
    ]
    result["preferred_direction_sign_at_decision"] = decision[
        "preferred_direction_sign"
    ].astype(int)

    decision_sign = result["preferred_direction_sign_at_decision"].astype(float)
    result["potential_descent_speed_at_decision_per_sd"] = -decision_sign * decision[
        "potential_derivative_median_per_sd"
    ]
    result["potential_descent_speed_at_previous_same_direction_per_sd"] = (
        -decision_sign * previous["potential_derivative_median_per_sd"]
    )
    result["signed_speed_change_previous_to_decision_per_sd"] = (
        result["potential_descent_speed_at_decision_per_sd"]
        - result["potential_descent_speed_at_previous_same_direction_per_sd"]
    )
    result["absolute_speed_change_previous_to_decision_per_sd"] = result[
        "signed_speed_change_previous_to_decision_per_sd"
    ].abs()
    signed_change = result["signed_speed_change_previous_to_decision_per_sd"]
    result["speed_change_direction_at_decision"] = np.select(
        [signed_change > 0.0, signed_change < 0.0],
        ["faster", "slower"],
        default="unchanged",
    )

    result["preferred_descent_fraction_at_decision"] = decision[
        "preferred_descent_fraction"
    ]
    same_direction = (
        decision["preferred_direction_sign"].astype(int)
        == previous["preferred_direction_sign"].astype(int)
    )
    result["same_preferred_direction_between_times"] = same_direction
    result["descent_fraction_at_previous_along_decision_direction"] = np.where(
        same_direction,
        previous["preferred_descent_fraction"],
        1.0 - previous["preferred_descent_fraction"],
    )
    result["direction_consistent_80_at_decision"] = (
        result["preferred_descent_fraction_at_decision"] >= 0.80
    )
    result["direction_consistent_90_at_decision"] = (
        result["preferred_descent_fraction_at_decision"] >= 0.90
    )

    absolute_order = result.sort_values(
        [
            "potential_descent_speed_at_decision_per_sd",
            "preferred_descent_fraction_at_decision",
        ],
        ascending=[False, False],
        kind="mergesort",
    ).index
    result["decision_time_absolute_speed_rank"] = pd.Series(
        np.arange(1, len(result) + 1), index=absolute_order
    )
    result = result.sort_values(
        [
            "absolute_speed_change_previous_to_decision_per_sd",
            "potential_descent_speed_at_decision_per_sd",
            "preferred_descent_fraction_at_decision",
        ],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index()
    result.insert(1, "absolute_speed_change_rank", np.arange(1, len(result) + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_tag = f"{time_tag(args.decision_time)}_vs_{time_tag(args.previous_time)}"
    ranking_path = args.output_dir / f"hvg2000_{comparison_tag}_speed_change_ranking.csv"
    focus_path = args.output_dir / f"focus_genes_{comparison_tag}_speed_change.csv"
    summary_path = args.output_dir / f"{comparison_tag}_speed_change_summary.json"
    readme_path = args.output_dir / "README.md"
    result.to_csv(ranking_path, index=False)

    focus = result.loc[result["gene"].isin(FOCUS_GENES)].copy()
    if args.finite_check is not None and args.finite_check.exists():
        finite = pd.read_csv(args.finite_check)
        finite = finite.loc[
            np.isclose(finite["time"], args.decision_time)
            & finite["gene"].isin(FOCUS_GENES)
        ].copy()
        finite = finite.rename(
            columns={
                "preferred_expression_direction": "finite_check_direction",
                "delta_potential_mean": "finite_delta_potential_mean",
                "delta_potential_median": "finite_delta_potential_median",
                "fraction_potential_decreased": "finite_fraction_decreased",
            }
        )
        finite_columns = [
            "gene",
            "finite_check_direction",
            "finite_dose_sd",
            "finite_delta_potential_mean",
            "finite_delta_potential_median",
            "finite_fraction_decreased",
        ]
        focus = focus.merge(finite.loc[:, finite_columns], on="gene", how="left")
        focus["finite_check_matches_decision_direction"] = (
            focus["finite_check_direction"]
            == focus["preferred_expression_direction_at_decision"]
        )
    focus.to_csv(focus_path, index=False)

    summary = {
        "status": "DONE",
        "decision_time": float(args.decision_time),
        "previous_time": float(args.previous_time),
        "hvg_count": int(len(result)),
        "definition": (
            "Choose each gene's expression increase/decrease direction from the "
            "population-median potential gradient at the decision time. Evaluate "
            "potential-descent speed along that same direction at both times."
        ),
        "primary_score": (
            "Absolute decision-minus-previous change in median potential-descent "
            "speed per one observed-expression SD."
        ),
        "scope": "All 2000 HVGs; no DNB membership or top-k pool is used.",
        "faster_at_decision_count": int((signed_change > 0.0).sum()),
        "slower_at_decision_count": int((signed_change < 0.0).sum()),
        "direction_consistent_80_count": int(
            result["direction_consistent_80_at_decision"].sum()
        ),
        "same_direction_count": int(
            result["same_preferred_direction_between_times"].sum()
        ),
        "focus_genes": focus.to_dict(orient="records"),
        "inputs": {
            "potential_by_time": str(args.input.resolve()),
            "finite_difference_check": (
                None
                if args.finite_check is None
                else str(args.finite_check.resolve())
            ),
        },
        "outputs": {
            "ranking": str(ranking_path.resolve()),
            "focus_genes": str(focus_path.resolve()),
            "summary": str(summary_path.resolve()),
            "readme": str(readme_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(
        f"# E{args.decision_time:.1f} versus E{args.previous_time:.1f} potential speed change\n\n"
        "All 2000 HVGs are ranked by the absolute change in local potential-descent "
        "speed between the adjacent reconstructed times. Both faster and slower "
        "changes are retained. The expression direction is fixed at the decision "
        "time and evaluated at both times. DNB membership is not used.\n",
        encoding="utf-8",
    )
    print(f"Wrote {ranking_path}")
    print(f"Wrote {focus_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rank HVGs by median cell-wise absolute gene-direction potential gradient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOCUS_GENES = ("Glis3", "Chd7", "Fzd3", "Efna5")
SPEED_COLUMN = "median_abs_potential_derivative_per_sd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--times", nargs="+", type=float, required=True)
    return parser.parse_args()


def time_tag(time: float) -> str:
    return f"e{time:.1f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    if SPEED_COLUMN not in source.columns:
        raise ValueError(
            f"Missing {SPEED_COLUMN}; rerun the model-gradient calculation with the "
            "updated summarizer."
        )

    ranked_parts: list[pd.DataFrame] = []
    requested_times = list(map(float, args.times))
    for time in requested_times:
        current = source.loc[np.isclose(source["time"], time)].copy()
        if current.empty:
            raise ValueError(f"No rows found at time {time:g}")
        if current["gene"].duplicated().any():
            raise ValueError(f"Duplicate genes found at time {time:g}")
        current = current.sort_values(
            [SPEED_COLUMN, "mean_abs_potential_derivative_per_sd", "gene"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        current.insert(1, "absolute_potential_speed_rank", np.arange(1, len(current) + 1))
        current.insert(
            2,
            "absolute_potential_speed_top_fraction",
            current["absolute_potential_speed_rank"] / float(len(current)),
        )
        current.insert(
            3,
            "within_top_20_percent",
            current["absolute_potential_speed_rank"] <= int(np.floor(0.20 * len(current))),
        )
        ranked_parts.append(current)

    ranking = pd.concat(ranked_parts, ignore_index=True)
    focus = ranking.loc[ranking["gene"].isin(FOCUS_GENES)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(map(time_tag, requested_times))
    ranking_path = args.output_dir / f"hvg2000_absolute_potential_speed_{tag}.csv"
    focus_path = args.output_dir / f"focus_genes_absolute_potential_speed_{tag}.csv"
    summary_path = args.output_dir / f"absolute_potential_speed_{tag}_summary.json"
    readme_path = args.output_dir / "README.md"
    ranking.to_csv(ranking_path, index=False)
    focus.to_csv(focus_path, index=False)

    summary = {
        "status": "DONE",
        "times": requested_times,
        "hvg_count_per_time": {
            f"{time:g}": int((ranking["time"] == time).sum()) for time in requested_times
        },
        "definition": (
            "For each cell and gene, take the absolute value of the gradient of "
            "U(s, encoder(expression), t) with respect to that gene, scaled by one "
            "observed-expression SD. The gene score is the median absolute gradient "
            "across cells at that time."
        ),
        "interpretation": (
            "Each cell may use its own downhill expression sign. The score measures "
            "gene-axis potential sensitivity, not a shared overexpression/knockdown "
            "direction, not dU/dt, and not experimental causality."
        ),
        "scope": "All 2000 HVGs; no DNB, top-k pool, or direction-consistency filter.",
        "focus_genes": focus.to_dict(orient="records"),
        "inputs": {"potential_by_time": str(args.input.resolve())},
        "outputs": {
            "ranking": str(ranking_path.resolve()),
            "focus_genes": str(focus_path.resolve()),
            "summary": str(summary_path.resolve()),
            "readme": str(readme_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(
        "# Absolute gene-direction potential speed\n\n"
        "All 2000 HVGs are ranked independently at each requested time by the median "
        "across cells of the absolute one-expression-SD potential gradient. Each cell "
        "may use its own downhill expression sign. No DNB membership, candidate pool, "
        "or direction-consistency filter is used.\n",
        encoding="utf-8",
    )
    print(f"Wrote {ranking_path}")
    print(f"Wrote {focus_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

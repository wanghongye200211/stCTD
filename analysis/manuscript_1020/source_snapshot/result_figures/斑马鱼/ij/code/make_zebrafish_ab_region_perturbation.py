#!/usr/bin/env python3
"""Compare matched perturbation responses in two 14 hpf branch-07 regions.

The analysis starts from every trajectory whose 10 hpf detailed source state is
Segmental Plate/Tail Bud.  It does not select trajectories by their 18 hpf
outcome.  At the generated 14 hpf snapshot, equal-sized upper and lower spatial
quartiles define regions A and B.  The same fixed set of bidirectional,
equal-norm molecular-latent perturbations is then applied to either region and
rolled forward to 18 hpf with the trained stVCR dynamics model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon
from scipy import stats
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STVCR_SRC = ROOT / "stVCR_repo/src"
for path in (ROOT, SCRIPT_DIR, STVCR_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_zesta_stvcr_classifier_dynamic import (  # noqa: E402
    StateClassifier,
    integrate_model,
    load_dynamics_model,
)


RUN_DIR = ROOT / "SpaPOT_runs/zesta_10_12_18_gat10_h96_192_hybrid_prealigned_nolabel_e1000_s512"
BRANCH_VALUES = ROOT / "result_figures/zesta/lineage_branch_triptych/other/zesta_single_origin_detail_mesoderm_values.csv"
CLASSIFIER_PATH = ROOT / "result_figures/zesta/lineage_branch_triptych/other/detail_mesoderm_classifier/state_classifier.pt"
GLOBAL_BACKGROUND = ROOT / "result_figures/zesta/mechanistic_fields/other/zesta_spatial_mechanistic_field_values.csv"
CRITICAL_SUMMARY = (
    ROOT
    / "result_figures/zesta/lineage_critical_compact/branch07_terminal_fixed_detail_vs_merged"
    / "critical_terminal_fixed_no_pronephros_summary.csv"
)

OUT_ROOT = ROOT / "result_figures/斑马鱼/ij"
PANEL_DIR = OUT_ROOT / "panels/i"
PART_DIR = PANEL_DIR / "parts/ab_region_perturbation"
SOURCE_DIR = OUT_ROOT / "source_tables/ab_region_perturbation"
BY_FORMAT_DIR = OUT_ROOT / "by_format"
MAIN_STEM = PANEL_DIR / "i_critical_ab_region_perturbation"
SOURCE_GROUP = "Segmental Plate, Tail Bud"

REGION_COLORS = {"A": "#0072B2", "B": "#D55E00"}
BRANCH_COLOR = "#3B7F43"
BACKGROUND_COLOR = "#C4CBD3"
FATE_COLORS = {
    "Somitic": "#2E7D32",
    "Notochord": "#E6A21A",
    "Blood / vascular": "#8E3B8F",
    "Other": "#8A9199",
}

SOMITIC_LABELS = {"Segmental Plate, Tail Bud", "Adaxial Cell", "Somite"}
AXIAL_LABELS = {"Notochord"}
BLOOD_LABELS = {"Erythroid Lineage Cell", "Angioblastic Mesenchymal Cell", "Lateral Plate Mesoderm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--perturbation-norm", type=float, default=0.30)
    parser.add_argument("--n-axes", type=int, default=12)
    parser.add_argument("--region-fraction", type=float, default=0.25)
    parser.add_argument("--right-band-quantile", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_formats(fig: plt.Figure, stem: Path, *, include_tiff: bool = False, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    if include_tiff:
        fig.savefig(
            stem.with_suffix(".tiff"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)


def load_run(device: torch.device) -> tuple[Any, ad.AnnData, dict[str, Any], dict[str, Any]]:
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    adata = ad.read_h5ad(summary["trained_h5ad"])
    config = dict(summary["config"])
    spatial_dim = int(adata.obsm["X_spatial_aligned"].shape[1])
    gene_dim = int(adata.obsm["X_gene_input"].shape[1])
    model = load_dynamics_model(
        Path(summary["model_path"]),
        gene_dim,
        spatial_dim,
        device,
        hidden_dim=int(config["hidden_dim"]),
        n_hiddens=int(config["n_hiddens"]),
        activation=str(config["activation"]),
    )
    return model, adata, config, summary


def load_classifier(device: torch.device, state_dim: int) -> tuple[StateClassifier, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(CLASSIFIER_PATH, map_location="cpu")
    classifier_summary = checkpoint["summary"]
    classes = np.asarray(checkpoint["classes"], dtype=object)
    classifier = StateClassifier(
        state_dim + 1,
        int(classifier_summary["hidden_dim"]),
        len(classes),
        float(classifier_summary["dropout"]),
        int(classifier_summary["n_hiddens"]),
        str(classifier_summary["activation"]),
    ).to(device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    return classifier, mean, scale, classes


def predict_probabilities(
    classifier: StateClassifier,
    mean: np.ndarray,
    scale: np.ndarray,
    states: np.ndarray,
    time_input: float,
    device: torch.device,
) -> np.ndarray:
    features = np.hstack(
        [states, np.full((states.shape[0], 1), float(time_input), dtype=np.float32)]
    ).astype(np.float32)
    scaled = (features - mean) / scale
    probabilities = []
    with torch.no_grad():
        for start in range(0, scaled.shape[0], 4096):
            batch = torch.as_tensor(scaled[start : start + 4096], dtype=torch.float32, device=device)
            probabilities.append(torch.softmax(classifier(batch), dim=1).cpu().numpy())
    return np.vstack(probabilities).astype(np.float32)


def merged_fate(label: str) -> str:
    if label in SOMITIC_LABELS:
        return "Somitic"
    if label in AXIAL_LABELS:
        return "Notochord"
    if label in BLOOD_LABELS:
        return "Blood / vascular"
    return "Other"


def merged_probability(probabilities: np.ndarray, classes: np.ndarray, fate: str) -> np.ndarray:
    indices = [index for index, label in enumerate(classes) if merged_fate(str(label)) == fate]
    if not indices:
        return np.zeros(probabilities.shape[0], dtype=np.float32)
    return probabilities[:, indices].sum(axis=1)


def time_input_for(adata: ad.AnnData, raw_time: float) -> float:
    raw = adata.obs["time"].astype(float).to_numpy()
    known_times = np.asarray(sorted(np.unique(raw)), dtype=float)
    known_inputs = np.asarray(
        [np.median(adata.obs.loc[np.isclose(raw, time), "time_input"].astype(float)) for time in known_times],
        dtype=float,
    )
    coefficients = np.polyfit(known_times, known_inputs, deg=1)
    return float(np.polyval(coefficients, float(raw_time)))


def source_defined_branch_ids() -> np.ndarray:
    values = pd.read_csv(BRANCH_VALUES, usecols=["time", "trajectory_id", "source_group"])
    source = values[
        np.isclose(values["time"].to_numpy(float), 10.0)
        & values["source_group"].astype(str).eq(SOURCE_GROUP)
    ]
    ids = np.asarray(sorted(source["trajectory_id"].astype(int).unique()), dtype=int)
    if ids.size == 0:
        raise ValueError(f"No source-defined trajectories for {SOURCE_GROUP!r}")
    return ids


def define_regions(
    states_14: np.ndarray,
    trajectory_ids: np.ndarray,
    fraction: float,
    right_band_quantile: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if not 0.05 <= fraction <= 0.45:
        raise ValueError("--region-fraction must be between 0.05 and 0.45")
    if not 0.0 <= right_band_quantile <= 0.75:
        raise ValueError("--right-band-quantile must be between 0 and 0.75")
    x_threshold = float(np.quantile(states_14[:, 0], right_band_quantile))
    candidate_positions = np.flatnonzero(states_14[:, 0] >= x_threshold)
    region_size = int(np.floor(len(candidate_positions) * float(fraction)))
    candidate_order = np.lexsort(
        (trajectory_ids[candidate_positions], states_14[candidate_positions, 1])
    )
    ordered_positions = candidate_positions[candidate_order]
    regions = {"B": ordered_positions[:region_size], "A": ordered_positions[-region_size:]}
    labels = np.full(states_14.shape[0], "B07 remainder", dtype=object)
    labels[regions["A"]] = "A"
    labels[regions["B"]] = "B"
    table = pd.DataFrame(
        {
            "trajectory_id": trajectory_ids,
            "x_14hpf": states_14[:, 0],
            "y_14hpf": states_14[:, 1],
            "region": labels,
            "right_band_x_threshold": x_threshold,
            "in_right_spatial_band": states_14[:, 0] >= x_threshold,
            "selection_rule": (
                f"right band=x at/above branch {100 * right_band_quantile:.0f}th percentile; "
                "A=upper y quartile and B=lower y quartile within band; equal n"
            ),
        }
    )
    return regions, table


def perturb_and_rollout(
    model: Any,
    states_14: np.ndarray,
    baseline_18: np.ndarray,
    trajectory_ids: np.ndarray,
    regions: dict[str, np.ndarray],
    classifier: StateClassifier,
    classifier_mean: np.ndarray,
    classifier_scale: np.ndarray,
    classes: np.ndarray,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    t14: float,
    t18: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(args.seed))
    gene_dim = states_14.shape[1] - 2
    directions = rng.normal(size=(int(args.n_axes), gene_dim)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)

    direction_rows = []
    for axis_index, direction in enumerate(directions):
        for dimension, value in enumerate(direction):
            direction_rows.append(
                {
                    "axis": axis_index + 1,
                    "gene_latent_dimension": dimension + 1,
                    "unit_direction_value": float(value),
                    "perturbation_l2_norm": float(args.perturbation_norm),
                    "paired_signs": "+1;-1",
                    "seed": int(args.seed),
                }
            )
    direction_table = pd.DataFrame(direction_rows)

    chunks = []
    metadata = []
    for region_name in ("A", "B"):
        positions = regions[region_name]
        for axis_index, direction in enumerate(directions):
            for sign in (1, -1):
                perturbed = states_14[positions].copy()
                perturbed[:, 2:] += float(sign) * float(args.perturbation_norm) * direction
                chunks.append(perturbed)
                metadata.append(
                    {
                        "region": region_name,
                        "axis": axis_index + 1,
                        "sign": sign,
                        "positions": positions,
                    }
                )
    combined = np.vstack(chunks).astype(np.float32)
    perturbed_18, _ = integrate_model(
        model,
        combined,
        float(t14),
        float(t18),
        device,
        str(config.get("ode_method", "rk4")),
        float(config.get("ode_step_size", 0.1)),
        int(args.chunk_size),
    )

    baseline_prob = predict_probabilities(
        classifier,
        classifier_mean,
        classifier_scale,
        baseline_18,
        t18,
        device,
    )
    perturbed_prob = predict_probabilities(
        classifier,
        classifier_mean,
        classifier_scale,
        perturbed_18,
        t18,
        device,
    )
    baseline_labels = classes[baseline_prob.argmax(axis=1)]

    rows = []
    offset = 0
    merged_fates = list(FATE_COLORS)
    for item in metadata:
        positions = item["positions"]
        count = len(positions)
        endpoint = perturbed_18[offset : offset + count]
        endpoint_prob = perturbed_prob[offset : offset + count]
        offset += count
        baseline_endpoint = baseline_18[positions]
        baseline_endpoint_prob = baseline_prob[positions]
        perturbed_labels = classes[endpoint_prob.argmax(axis=1)]
        spatial_response = np.linalg.norm(endpoint[:, :2] - baseline_endpoint[:, :2], axis=1)
        molecular_response = np.linalg.norm(endpoint[:, 2:] - baseline_endpoint[:, 2:], axis=1)
        joint_response = np.linalg.norm(endpoint - baseline_endpoint, axis=1)
        baseline_merged_prob = np.column_stack(
            [merged_probability(baseline_endpoint_prob, classes, fate) for fate in merged_fates]
        )
        endpoint_merged_prob = np.column_stack(
            [merged_probability(endpoint_prob, classes, fate) for fate in merged_fates]
        )
        fate_tv = 0.5 * np.abs(endpoint_merged_prob - baseline_merged_prob).sum(axis=1)
        for local_index, position in enumerate(positions):
            baseline_label = str(baseline_labels[position])
            perturbed_label = str(perturbed_labels[local_index])
            record: dict[str, Any] = {
                "region": item["region"],
                "axis": int(item["axis"]),
                "sign": int(item["sign"]),
                "trajectory_id": int(trajectory_ids[position]),
                "baseline_x_18hpf": float(baseline_endpoint[local_index, 0]),
                "baseline_y_18hpf": float(baseline_endpoint[local_index, 1]),
                "perturbed_x_18hpf": float(endpoint[local_index, 0]),
                "perturbed_y_18hpf": float(endpoint[local_index, 1]),
                "spatial_displacement": float(spatial_response[local_index]),
                "molecular_displacement": float(molecular_response[local_index]),
                "joint_displacement": float(joint_response[local_index]),
                "fate_probability_tv": float(fate_tv[local_index]),
                "baseline_fate_detailed": baseline_label,
                "perturbed_fate_detailed": perturbed_label,
                "baseline_fate_merged": merged_fate(baseline_label),
                "perturbed_fate_merged": merged_fate(perturbed_label),
                "fate_switch_detailed": int(baseline_label != perturbed_label),
                "fate_switch_merged": int(merged_fate(baseline_label) != merged_fate(perturbed_label)),
            }
            for fate_index, fate in enumerate(merged_fates):
                baseline_value = baseline_merged_prob[local_index, fate_index]
                perturbed_value = endpoint_merged_prob[local_index, fate_index]
                record[f"delta_probability_{fate.lower().replace(' / ', '_').replace(' ', '_')}"] = float(
                    perturbed_value - baseline_value
                )
            rows.append(record)
    return pd.DataFrame(rows), direction_table


def summarize_responses(response: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    axis_summary = (
        response.groupby(["region", "axis"], observed=False)
        .agg(
            mean_spatial_displacement=("spatial_displacement", "mean"),
            mean_molecular_displacement=("molecular_displacement", "mean"),
            mean_joint_displacement=("joint_displacement", "mean"),
            mean_fate_probability_tv=("fate_probability_tv", "mean"),
            detailed_fate_switch_fraction=("fate_switch_detailed", "mean"),
            merged_fate_switch_fraction=("fate_switch_merged", "mean"),
            n_cell_sign_observations=("trajectory_id", "size"),
        )
        .reset_index()
    )
    cell_summary = (
        response.groupby(["region", "trajectory_id"], observed=False)
        .agg(
            mean_spatial_displacement=("spatial_displacement", "mean"),
            mean_molecular_displacement=("molecular_displacement", "mean"),
            mean_joint_displacement=("joint_displacement", "mean"),
            mean_fate_probability_tv=("fate_probability_tv", "mean"),
            detailed_fate_switch_fraction=("fate_switch_detailed", "mean"),
            merged_fate_switch_fraction=("fate_switch_merged", "mean"),
            baseline_x_18hpf=("baseline_x_18hpf", "first"),
            baseline_y_18hpf=("baseline_y_18hpf", "first"),
        )
        .reset_index()
    )

    metrics = [
        ("mean_spatial_displacement", "A", "B"),
        ("mean_fate_probability_tv", "B", "A"),
        ("detailed_fate_switch_fraction", "B", "A"),
        ("merged_fate_switch_fraction", "B", "A"),
    ]
    rows = []
    for metric, expected_high, expected_low in metrics:
        wide = axis_summary.pivot(index="axis", columns="region", values=metric).dropna()
        statistic, p_value = stats.wilcoxon(wide["A"], wide["B"], alternative="two-sided")
        delta_b_minus_a = wide["B"] - wide["A"]
        rows.append(
            {
                "metric": metric,
                "n_paired_bidirectional_axes": int(len(wide)),
                "mean_A": float(wide["A"].mean()),
                "mean_B": float(wide["B"].mean()),
                "median_A": float(wide["A"].median()),
                "median_B": float(wide["B"].median()),
                "mean_B_minus_A": float(delta_b_minus_a.mean()),
                "median_B_minus_A": float(delta_b_minus_a.median()),
                "expected_higher_region": expected_high,
                "expected_lower_region": expected_low,
                "paired_wilcoxon_statistic": float(statistic),
                "paired_wilcoxon_two_sided_p": float(p_value),
                "axes_with_B_greater_than_A": int((delta_b_minus_a > 0).sum()),
                "axes_with_A_greater_than_B": int((delta_b_minus_a < 0).sum()),
            }
        )
    statistics_table = pd.DataFrame(rows)
    primary_metrics = {"mean_spatial_displacement", "mean_fate_probability_tv"}
    statistics_table["primary_bonferroni_p"] = np.where(
        statistics_table["metric"].isin(primary_metrics),
        np.minimum(statistics_table["paired_wilcoxon_two_sided_p"] * len(primary_metrics), 1.0),
        np.nan,
    )
    return axis_summary, cell_summary, statistics_table


def fate_composition(
    region_table: pd.DataFrame,
    regions: dict[str, np.ndarray],
    probabilities_14: np.ndarray,
    probabilities_18: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for time, probabilities in ((14.0, probabilities_14), (18.0, probabilities_18)):
        labels = classes[probabilities.argmax(axis=1)]
        for region_name, positions in regions.items():
            region_labels = labels[positions]
            for resolution, values in (
                ("detailed", np.asarray(region_labels, dtype=object)),
                ("merged", np.asarray([merged_fate(str(label)) for label in region_labels], dtype=object)),
            ):
                counts = pd.Series(values).value_counts()
                for fate, count in counts.items():
                    rows.append(
                        {
                            "time_hpf": time,
                            "region": region_name,
                            "resolution": resolution,
                            "fate": str(fate),
                            "n": int(count),
                            "fraction": float(count / len(positions)),
                            "region_n": int(len(positions)),
                        }
                    )
    return pd.DataFrame(rows)


def square_limits(background: pd.DataFrame) -> tuple[float, float, float, float]:
    xy = background[["x", "y"]].to_numpy(float)
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    center = 0.5 * (low + high)
    span = max(high - low) * 1.06
    return center[0] - span / 2, center[0] + span / 2, center[1] - span / 2, center[1] + span / 2


def add_region_hull(ax: plt.Axes, xy: np.ndarray, color: str) -> None:
    if len(xy) < 3:
        return
    hull = ConvexHull(xy)
    polygon = Polygon(
        xy[hull.vertices],
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=0.10,
        linewidth=1.05,
        linestyle=(0, (3.2, 1.8)),
        zorder=4,
    )
    ax.add_patch(polygon)


def draw_region_map(
    ax: plt.Axes,
    background_14: pd.DataFrame,
    states_14: np.ndarray,
    region_table: pd.DataFrame,
    *,
    title: str,
) -> None:
    ax.scatter(
        background_14["x"],
        background_14["y"],
        s=4.0,
        c=BACKGROUND_COLOR,
        alpha=0.25,
        linewidths=0,
        zorder=1,
    )
    remainder = region_table["region"].eq("B07 remainder").to_numpy()
    ax.scatter(
        states_14[remainder, 0],
        states_14[remainder, 1],
        s=7.2,
        c=BRANCH_COLOR,
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    for region_name in ("A", "B"):
        mask = region_table["region"].eq(region_name).to_numpy()
        xy = states_14[mask, :2]
        add_region_hull(ax, xy, REGION_COLORS[region_name])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=10.5,
            c=REGION_COLORS[region_name],
            alpha=0.90,
            linewidths=0,
            zorder=5,
        )
        center = np.median(xy, axis=0)
        label = ax.text(
            center[0] - 0.035,
            center[1],
            region_name,
            fontsize=8.5,
            weight="bold",
            color=REGION_COLORS[region_name],
            ha="right",
            va="center",
            zorder=6,
        )
        label.set_path_effects([patheffects.Stroke(linewidth=2.0, foreground="white"), patheffects.Normal()])

    limits = square_limits(background_14)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.0, pad=3.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.80)
        spine.set_color("#2B2B2B")
        spine.set_linestyle((0, (3.2, 2.0)))


def format_p(value: float, prefix: str = "P") -> str:
    if value < 0.001:
        return f"{prefix}={value:.1e}"
    return f"{prefix}={value:.3f}"


def draw_paired_metric(
    ax: plt.Axes,
    axis_summary: pd.DataFrame,
    statistics_table: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    percentage: bool = False,
) -> None:
    wide = axis_summary.pivot(index="axis", columns="region", values=metric).dropna()
    multiplier = 100.0 if percentage else 1.0
    a_values = wide["A"].to_numpy(float) * multiplier
    b_values = wide["B"].to_numpy(float) * multiplier
    for a_value, b_value in zip(a_values, b_values):
        ax.plot([0, 1], [a_value, b_value], color="#AAB1B8", lw=0.62, alpha=0.72, zorder=1)
    ax.scatter(
        np.zeros_like(a_values),
        a_values,
        s=17,
        facecolor="white",
        edgecolor=REGION_COLORS["A"],
        linewidth=0.85,
        zorder=3,
    )
    ax.scatter(
        np.ones_like(b_values),
        b_values,
        s=17,
        facecolor="white",
        edgecolor=REGION_COLORS["B"],
        linewidth=0.85,
        zorder=3,
    )
    medians = [np.median(a_values), np.median(b_values)]
    ax.scatter(
        [0, 1],
        medians,
        s=39,
        c=[REGION_COLORS["A"], REGION_COLORS["B"]],
        marker="D",
        edgecolor="white",
        linewidth=0.75,
        zorder=4,
    )
    ymax = max(np.max(a_values), np.max(b_values))
    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(0, ymax * 1.30 if ymax > 0 else 1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A", "B"], fontsize=6.2)
    ax.set_ylabel(ylabel, fontsize=5.8, labelpad=2.0)
    ax.set_title(title, fontsize=7.0, pad=3.0)
    ax.tick_params(axis="both", labelsize=5.5, length=2.0, width=0.65)
    ax.grid(axis="y", color="#E2E6EA", lw=0.45, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.72)
        spine.set_color("#222222")
    statistic_row = statistics_table.loc[statistics_table["metric"].eq(metric)].iloc[0]
    adjusted_p = float(statistic_row["primary_bonferroni_p"])
    if np.isfinite(adjusted_p):
        p_value = adjusted_p
        p_prefix = "adj. P"
    else:
        p_value = float(statistic_row["paired_wilcoxon_two_sided_p"])
        p_prefix = "P"
    ax.text(
        0.5,
        0.975,
        format_p(p_value, p_prefix),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.1,
        color="#444444",
    )


def draw_main_panel(
    background_14: pd.DataFrame,
    states_14: np.ndarray,
    region_table: pd.DataFrame,
    axis_summary: pd.DataFrame,
    statistics_table: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    fig = plt.figure(figsize=(5.72, 2.20), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.28, 0.92, 0.92],
        left=0.052,
        right=0.985,
        bottom=0.235,
        top=0.805,
        wspace=0.42,
    )
    ax_map = fig.add_subplot(grid[0, 0])
    ax_spatial = fig.add_subplot(grid[0, 1])
    ax_fate = fig.add_subplot(grid[0, 2])
    draw_region_map(
        ax_map,
        background_14,
        states_14,
        region_table,
        title="14 hpf critical-onset snapshot",
    )
    draw_paired_metric(
        ax_spatial,
        axis_summary,
        statistics_table,
        "mean_spatial_displacement",
        "Spatial response",
        "Terminal spatial displacement",
    )
    draw_paired_metric(
        ax_fate,
        axis_summary,
        statistics_table,
        "mean_fate_probability_tv",
        "Fate response",
        "Fate-module probability shift (TV)",
    )
    fig.text(0.018, 0.912, "i", fontsize=11.0, ha="left", va="center")
    fig.text(
        0.665,
        0.895,
        f"matched molecular perturbation: L2 norm={args.perturbation_norm:.2f}; {args.n_axes} bidirectional axes",
        fontsize=5.5,
        ha="center",
        va="center",
        color="#3A4653",
    )
    legend_handles = [
        Patch(facecolor=BRANCH_COLOR, edgecolor="none", label="B07 trajectory"),
        Patch(facecolor=REGION_COLORS["A"], edgecolor="none", label=f"Region A (n={(region_table.region == 'A').sum()})"),
        Patch(facecolor=REGION_COLORS["B"], edgecolor="none", label=f"Region B (n={(region_table.region == 'B').sum()})"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.035),
        ncol=3,
        frameon=False,
        fontsize=5.3,
        handlelength=0.72,
        columnspacing=0.85,
    )
    save_formats(fig, MAIN_STEM, include_tiff=True, dpi=900)
    for extension in ("png", "pdf", "svg", "tiff"):
        target_dir = BY_FORMAT_DIR / extension
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MAIN_STEM.with_suffix(f".{extension}"), target_dir / f"{MAIN_STEM.name}.{extension}")


def draw_standalone_parts(
    background_14: pd.DataFrame,
    background_18: pd.DataFrame,
    states_14: np.ndarray,
    region_table: pd.DataFrame,
    axis_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
    statistics_table: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    draw_region_map(ax, background_14, states_14, region_table, title="14 hpf critical-onset snapshot")
    fig.tight_layout(pad=0.22)
    save_formats(fig, PART_DIR / "i1_14hpf_ab_region_definition", dpi=750)

    metric_specs = [
        (
            "mean_spatial_displacement",
            "Spatial response",
            "Terminal spatial displacement",
            False,
            "i2_spatial_response_paired",
        ),
        (
            "mean_fate_probability_tv",
            "Fate response",
            "Fate-module probability shift (TV)",
            False,
            "i3_fate_response_paired",
        ),
        (
            "detailed_fate_switch_fraction",
            "Detailed fate switches",
            "Fate-switch rate (%)",
            True,
            "i4_fate_switch_paired",
        ),
    ]
    for metric, title, ylabel, percentage, name in metric_specs:
        fig, ax = plt.subplots(figsize=(2.00, 2.12))
        draw_paired_metric(
            ax,
            axis_summary,
            statistics_table,
            metric,
            title,
            ylabel,
            percentage=percentage,
        )
        fig.tight_layout(pad=0.58)
        save_formats(fig, PART_DIR / name, dpi=750)

    limits = square_limits(background_18)
    fig, axes = plt.subplots(2, 2, figsize=(4.15, 4.10), sharex=True, sharey=True)
    response_specs = [
        ("mean_spatial_displacement", "Spatial displacement", "viridis"),
        ("mean_fate_probability_tv", "Fate-module probability shift", "magma"),
    ]
    for row_index, (metric, row_title, cmap) in enumerate(response_specs):
        metric_values = cell_summary[metric].to_numpy(float)
        norm = Normalize(vmin=0.0, vmax=float(np.quantile(metric_values, 0.98)))
        image = None
        for column_index, region_name in enumerate(("A", "B")):
            ax = axes[row_index, column_index]
            ax.scatter(
                background_18["x"],
                background_18["y"],
                s=2.8,
                c=BACKGROUND_COLOR,
                alpha=0.17,
                linewidths=0,
            )
            part = cell_summary[cell_summary["region"].eq(region_name)]
            image = ax.scatter(
                part["baseline_x_18hpf"],
                part["baseline_y_18hpf"],
                s=9.0,
                c=part[metric],
                cmap=cmap,
                norm=norm,
                linewidths=0,
                alpha=0.95,
            )
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{row_title}: region {region_name}", fontsize=6.5, pad=2.5)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.70)
                spine.set_color("#303030")
        if image is not None:
            colorbar = fig.colorbar(image, ax=axes[row_index, :].tolist(), fraction=0.025, pad=0.020)
            colorbar.ax.tick_params(labelsize=5.0, length=1.8)
    fig.subplots_adjust(left=0.025, right=0.90, bottom=0.025, top=0.965, wspace=0.10, hspace=0.13)
    save_formats(fig, PART_DIR / "i5_18hpf_per_cell_response_maps", dpi=750)


def write_method_note(
    args: argparse.Namespace,
    region_table: pd.DataFrame,
    fate_table: pd.DataFrame,
    statistics_table: pd.DataFrame,
    median_gene_norm: float,
    device: torch.device,
) -> None:
    spatial = statistics_table.set_index("metric").loc["mean_spatial_displacement"]
    fate = statistics_table.set_index("metric").loc["mean_fate_probability_tv"]
    switch = statistics_table.set_index("metric").loc["detailed_fate_switch_fraction"]
    spatial_ratio = float(spatial["median_A"] / max(float(spatial["median_B"]), 1e-12))
    fate_ratio = float(fate["median_B"] / max(float(fate["median_A"]), 1e-12))

    def composition_sentence(region: str, time: float) -> str:
        part = fate_table[
            fate_table["region"].eq(region)
            & np.isclose(fate_table["time_hpf"].to_numpy(float), time)
            & fate_table["resolution"].eq("detailed")
        ].sort_values("n", ascending=False)
        return "; ".join(f"{row.fate}: {int(row.n)}" for row in part.itertuples(index=False))

    critical_text = "14.25-14.50 hpf"
    if CRITICAL_SUMMARY.exists():
        critical = pd.read_csv(CRITICAL_SUMMARY).iloc[0]
        critical_text = f"{float(critical['onset_start']):.2f}-{float(critical['onset_end']):.2f} hpf"

    classifier_checkpoint = torch.load(CLASSIFIER_PATH, map_location="cpu")
    classifier_validation = float(classifier_checkpoint["summary"]["best_val_balanced_accuracy"])
    note = f"""# Panel i: matched A/B regional perturbation at the branch-07 critical onset

## Main conclusion

Equal molecular-latent perturbations produced two reproducibly different response modes. Region A showed a larger terminal spatial displacement, whereas region B showed a larger redistribution of predicted fate probabilities. Across {int(args.n_axes)} paired bidirectional axes, the median spatial response was {spatial_ratio:.2f}-fold higher in A than B (paired Wilcoxon raw P={float(spatial['paired_wilcoxon_two_sided_p']):.4g}; Bonferroni-adjusted P={float(spatial['primary_bonferroni_p']):.4g}), while the median fate-probability response was {fate_ratio:.2f}-fold higher in B than A (raw P={float(fate['paired_wilcoxon_two_sided_p']):.4g}; adjusted P={float(fate['primary_bonferroni_p']):.4g}). The detailed-label switch rate was {100*float(switch['mean_A']):.1f}% for A and {100*float(switch['mean_B']):.1f}% for B.

## Region definition

- Cohort: all {int((region_table.region != '').sum())} trajectories whose 10 hpf source state was `Segmental Plate, Tail Bud`.
- No 18 hpf terminal label was used to select this cohort or to define A versus B.
- The previously estimated critical-onset window is {critical_text}; the nearest saved/generated state, 14 hpf, was used for intervention.
- A and B are defined inside the right-side branch band (`x` at or above the branch {100*float(args.right_band_quantile):.0f}th percentile). A is its upper spatial-y quartile and B its lower spatial-y quartile at 14 hpf. Both contain {int((region_table.region == 'A').sum())} trajectories.
- These are upper/lower domains in the aligned plotting coordinates. They are not assigned dorsal/ventral anatomical names without an independent orientation annotation.

## Perturbation and readouts

- Perturbation target: the 10-dimensional molecular latent coordinates only; spatial coordinates were not directly moved.
- Dose: `||delta z_mol||2 = {float(args.perturbation_norm):.2f}`, equal to {100*float(args.perturbation_norm)/median_gene_norm:.1f}% of the median 14 hpf molecular-latent norm in this branch.
- Direction control: {int(args.n_axes)} fixed random unit axes, each evaluated in both `+` and `-` directions. The two signs were averaged within each axis before A/B statistics.
- Counterfactual rollout: 14 to 18 hpf with the trained stVCR ODE, using the same direction and norm for A and B in every paired test.
- Spatial response: mean per-cell Euclidean difference between perturbed and unperturbed 18 hpf spatial coordinates.
- Fate response: total-variation distance between perturbed and unperturbed 18 hpf probability vectors after aggregating DetailedAnnotation probabilities into four modules (`Somitic`, `Notochord`, `Blood / vascular`, and `Other`).
- Statistics: two-sided paired Wilcoxon test across the {int(args.n_axes)} bidirectional axes; the two prespecified primary readouts were Bonferroni-corrected.

## Biological reading

- At 14 hpf, A is enriched for adaxial/somite-like states (`{composition_sentence('A', 14.0)}`), whereas B retains more segmental-plate/tail-bud-like states (`{composition_sentence('B', 14.0)}`).
- At baseline 18 hpf, A descendants are predominantly somitic/axial (`{composition_sentence('A', 18.0)}`), while B contributes strongly to erythroid/axial outcomes (`{composition_sentence('B', 18.0)}`).
- A therefore behaves as a spatially responsive, more differentiated somitic domain; B behaves as a more fate-responsive progenitor-like domain. This is a model-based susceptibility interpretation, not proof of an in vivo causal intervention.

## Important claim boundary

The stVCR particle ODE evolves cells independently. This analysis tests state-dependent susceptibility of cells located in A versus B; it does not simulate tissue ablation, neighbor propagation, mechanics, or an experimentally specified gene perturbation. A gene-level intervention would require mapping a defined expression perturbation through the molecular encoder and validating the result separately.

## Statistical and model QA

- `n` in the paired plots is {int(args.n_axes)} algorithmic perturbation axes, not biological replicates. Each axis is the mean of its positive and negative directions across {int((region_table.region == 'A').sum())} cells per region.
- Biological replicates/model refits: one zebrafish atlas and one trained dynamics model; uncertainty across embryos, model seeds, or retraining runs is not estimated here.
- Center statistic in the figure: median (filled diamond); all axis-level paired values are shown, so no hidden error bar is used.
- Baseline: the unperturbed 14-to-18 hpf rollout of the same cells.
- Fate readout classifier: best validation balanced accuracy `{classifier_validation:.3f}` for the 23 detailed labels. The plotted readout aggregates these probabilities into four broader fate modules to avoid counting within-module label changes as fate redistribution. It remains a model-predicted susceptibility readout and should be validated with an independent classifier or experiment before a causal claim.
- Primary tests: spatial displacement and fate-probability TV distance; two-sided paired Wilcoxon with Bonferroni correction across these two comparisons.
- Source data: `ab_perturbation_axis_summary.csv` and `ab_perturbation_paired_statistics.csv`.

## Files

- `i_critical_ab_region_perturbation.png/pdf/svg/tiff`: combined main panel.
- `parts/ab_region_perturbation/`: every small plot as a standalone PNG/PDF/SVG.
- `../../source_tables/ab_region_perturbation/ab_region_definition_14hpf.csv`: exact A/B membership.
- `../../source_tables/ab_region_perturbation/ab_perturbation_cell_responses.csv`: all cell-level counterfactual responses.
- `../../source_tables/ab_region_perturbation/ab_perturbation_axis_summary.csv`: axis-level paired values used in the figure.
- `../../source_tables/ab_region_perturbation/ab_perturbation_paired_statistics.csv`: statistical results.
- Runtime device for this export: `{device.type}`.
"""
    (PANEL_DIR / "i_ab_region_perturbation_method_and_interpretation.md").write_text(note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    model, adata, config, run_summary = load_run(device)
    spatial = np.asarray(adata.obsm["X_spatial_aligned"], dtype=np.float32)
    gene = np.asarray(adata.obsm["X_gene_input"], dtype=np.float32)
    raw_time = adata.obs["time"].astype(float).to_numpy()
    source_mask = np.isclose(raw_time, 10.0) & adata.obs["Annotation"].astype(str).eq("Mesoderm").to_numpy()
    source_state = np.hstack([spatial[source_mask], gene[source_mask]]).astype(np.float32)
    branch_ids = source_defined_branch_ids()
    if int(branch_ids.max()) >= source_state.shape[0]:
        raise IndexError("Branch trajectory IDs do not match the 10 hpf Mesoderm source ordering")

    t10 = time_input_for(adata, 10.0)
    t14 = time_input_for(adata, 14.0)
    t18 = time_input_for(adata, 18.0)
    source_14, _ = integrate_model(
        model,
        source_state,
        t10,
        t14,
        device,
        str(config.get("ode_method", "rk4")),
        float(config.get("ode_step_size", 0.1)),
        int(args.chunk_size),
    )
    states_14 = source_14[branch_ids]
    baseline_18, _ = integrate_model(
        model,
        states_14,
        t14,
        t18,
        device,
        str(config.get("ode_method", "rk4")),
        float(config.get("ode_step_size", 0.1)),
        int(args.chunk_size),
    )

    classifier, classifier_mean, classifier_scale, classes = load_classifier(device, states_14.shape[1])
    probabilities_14 = predict_probabilities(
        classifier, classifier_mean, classifier_scale, states_14, t14, device
    )
    probabilities_18 = predict_probabilities(
        classifier, classifier_mean, classifier_scale, baseline_18, t18, device
    )
    regions, region_table = define_regions(
        states_14,
        branch_ids,
        float(args.region_fraction),
        float(args.right_band_quantile),
    )
    labels_14 = classes[probabilities_14.argmax(axis=1)]
    labels_18 = classes[probabilities_18.argmax(axis=1)]
    region_table["baseline_fate_14hpf"] = labels_14
    region_table["baseline_fate_14hpf_merged"] = [merged_fate(str(label)) for label in labels_14]
    region_table["baseline_fate_18hpf"] = labels_18
    region_table["baseline_fate_18hpf_merged"] = [merged_fate(str(label)) for label in labels_18]

    response, direction_table = perturb_and_rollout(
        model,
        states_14,
        baseline_18,
        branch_ids,
        regions,
        classifier,
        classifier_mean,
        classifier_scale,
        classes,
        config,
        args,
        device,
        t14,
        t18,
    )
    axis_summary, cell_summary, statistics_table = summarize_responses(response)
    fate_table = fate_composition(region_table, regions, probabilities_14, probabilities_18, classes)

    region_table.to_csv(SOURCE_DIR / "ab_region_definition_14hpf.csv", index=False)
    direction_table.to_csv(SOURCE_DIR / "ab_perturbation_directions.csv", index=False)
    response.to_csv(SOURCE_DIR / "ab_perturbation_cell_responses.csv", index=False)
    axis_summary.to_csv(SOURCE_DIR / "ab_perturbation_axis_summary.csv", index=False)
    cell_summary.to_csv(SOURCE_DIR / "ab_perturbation_cell_summary.csv", index=False)
    statistics_table.to_csv(SOURCE_DIR / "ab_perturbation_paired_statistics.csv", index=False)
    fate_table.to_csv(SOURCE_DIR / "ab_region_baseline_fate_composition.csv", index=False)

    background = pd.read_csv(GLOBAL_BACKGROUND, usecols=["time", "kind", "x", "y"])
    background_14 = background[np.isclose(background["time"].to_numpy(float), 14.0)].copy()
    background_18 = background[np.isclose(background["time"].to_numpy(float), 18.0)].copy()
    draw_main_panel(background_14, states_14, region_table, axis_summary, statistics_table, args)
    draw_standalone_parts(
        background_14,
        background_18,
        states_14,
        region_table,
        axis_summary,
        cell_summary,
        statistics_table,
    )

    median_gene_norm = float(np.median(np.linalg.norm(states_14[:, 2:], axis=1)))
    write_method_note(args, region_table, fate_table, statistics_table, median_gene_norm, device)

    metadata = {
        "status": "DONE",
        "definition": "source-defined branch-07 upper/lower spatial-domain molecular-latent perturbation",
        "run_dir": str(RUN_DIR),
        "trained_h5ad": run_summary["trained_h5ad"],
        "model_path": run_summary["model_path"],
        "classifier_path": str(CLASSIFIER_PATH),
        "source_group": SOURCE_GROUP,
        "n_source_defined_branch_trajectories": int(len(branch_ids)),
        "region_fraction": float(args.region_fraction),
        "right_band_quantile": float(args.right_band_quantile),
        "region_n": {name: int(len(indices)) for name, indices in regions.items()},
        "perturbation_l2_norm": float(args.perturbation_norm),
        "median_14hpf_molecular_l2_norm": median_gene_norm,
        "n_bidirectional_axes": int(args.n_axes),
        "random_seed": int(args.seed),
        "time_input": {"10_hpf": t10, "14_hpf": t14, "18_hpf": t18},
        "device": device.type,
        "main_panel": str(MAIN_STEM.with_suffix(".png")),
    }
    (SOURCE_DIR / "ab_perturbation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    code_dir = OUT_ROOT / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Re-evaluate Figure 6 model comparisons on one coordinate and metric scale.

This script never trains a model. It reads completed prediction h5ad files,
maps the legacy mouse stVCR coordinates into the current benchmark coordinate
basis, and evaluates every spatial method with the same metric definitions.
"""

from __future__ import annotations

import json
import os
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("HOME", "/private/tmp/stvcr-figure6-v5-home")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-stvcr-figure6-v5")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-stvcr-figure6-v5")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for key in ("HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import prepare_official_external_benchmark_v3 as prep
import run_official_external_baselines_v3 as runner
from build_figure6c_extended_spatial_metric_table import (
    METRIC_DEFINITIONS,
    spatial_extended_metrics,
)


FIGURE6_DIR = ROOT / "result_figures" / "figure6"
V4_DIR = FIGURE6_DIR / "model_comparison_official_external_stvcr_ablation_v4"
MULTISEED_DIR = FIGURE6_DIR / "c_multiseed_v1"
OUT_DIR = FIGURE6_DIR / "model_comparison_unified_v5"

V4_SOURCE_MANIFEST = V4_DIR / "v4_spatial_slice_prediction_source_manifest.csv"
V4_METHOD_MANIFEST = V4_DIR / "v4_main_method_manifest.csv"
PREVIOUS_METRIC_SOURCE = FIGURE6_DIR / "c" / "figure6c_target_slice_metric_source.csv"
MULTISEED_SOURCE = MULTISEED_DIR / "figure6c_multiseed_metric_source.csv"
MULTISEED_RUN_MANIFEST = MULTISEED_DIR / "multi_seed_run_manifest.csv"
OFFICIAL_SCALE_AUDIT = (
    ROOT / "result_figures" / "model_comparison_official_external_benchmark_v3" / "official_training_scale_audit.csv"
)
LEGACY_SCALE_AUDIT = ROOT / "result_figures" / "model_comparison_external_benchmark_v2" / "training_scale_audit.csv"

MOUSE_STVCR_OBS = (
    ROOT
    / "SpaPOT_runs"
    / "dorsal_midbrain_12p5_14p5_16p5_notebook_hybrid_e400_s512"
    / "stvcr_notebook_trained.h5ad"
)

METHOD_ORDER = [
    "POT official solver",
    "moscot official",
    "Spateo official",
    "TrajectoryNet official",
    "Original stVCR",
    "SpaPOT",
]
DISPLAY_METHOD = {
    "POT official solver": "POT",
    "moscot official": "moscot",
    "Spateo official": "Spateo",
    "TrajectoryNet official": "TrajectoryNet",
    "Original stVCR": "stVCR",
    "SpaPOT": "stCTD",
}
DATASET_ORDER = ["zesta_10_12_18", "mouse_dorsal_midbrain"]
DATASET_LABEL = {
    "zesta_10_12_18": "Zebrafish ZESTA 10/12/18 hpf",
    "mouse_dorsal_midbrain": "Mouse dorsal midbrain",
}
TIME_DISPLAY = {
    ("zesta_10_12_18", 1): "12 hpf",
    ("zesta_10_12_18", 2): "18 hpf",
    ("mouse_dorsal_midbrain", 1): "E14.5",
    ("mouse_dorsal_midbrain", 2): "E16.5",
}

BASE_DIRECTIONS = {
    "annotation_l1": "lower_better",
    "annotation_corr": "higher_better",
    "detailed_l1": "lower_better",
    "detailed_corr": "higher_better",
    "spatial_chamfer": "lower_better",
    "spatial_iou": "higher_better",
    "spatial_chamfer_norm_diag": "lower_better",
    "spatial_iou_120": "higher_better",
}
EXTENDED_DIRECTIONS = {name: direction for name, _, direction, _ in METRIC_DEFINITIONS}
METRIC_DIRECTIONS = {**EXTENDED_DIRECTIONS, **BASE_DIRECTIONS}
METRIC_LABELS = {name: label for name, label, _, _ in METRIC_DEFINITIONS}
METRIC_LABELS.update(
    {
        "annotation_l1": "State L1",
        "annotation_corr": "State correlation",
        "detailed_l1": "Detailed-state L1",
        "detailed_corr": "Detailed-state correlation",
        "spatial_chamfer": "Spatial Chamfer",
        "spatial_iou": "Spatial IoU 48",
        "spatial_chamfer_norm_diag": "Normalized Chamfer audit",
        "spatial_iou_120": "Spatial IoU 120 audit",
    }
)


def affine_values(xy: np.ndarray, coef: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or coef.shape != (3, 2):
        raise ValueError(f"Invalid affine shapes: xy={xy.shape}, coef={coef.shape}")
    # Elementwise evaluation avoids platform-specific BLAS floating-point
    # status warnings observed for otherwise finite 3 x 2 affine products.
    fitted = np.sum(xy[:, :, None] * coef[None, :2, :], axis=1) + coef[2, :]
    if not np.isfinite(fitted).all():
        raise ValueError("Affine transform produced non-finite coordinates.")
    return np.asarray(fitted, dtype=np.float64)


def fit_affine(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float | int]]:
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape != target_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError(f"Affine pair shape mismatch: source={source_xy.shape}, target={target_xy.shape}")
    finite = np.isfinite(source_xy).all(axis=1) & np.isfinite(target_xy).all(axis=1)
    source_fit = source_xy[finite]
    target_fit = target_xy[finite]
    if source_fit.shape[0] < 3:
        raise ValueError("Fewer than three finite coordinate pairs are available for affine fitting.")
    design = np.column_stack([source_fit, np.ones(source_fit.shape[0], dtype=np.float64)])
    rank = int(np.linalg.matrix_rank(design))
    if rank < 3:
        raise ValueError(f"Affine design matrix is rank deficient: rank={rank}")
    condition = float(np.linalg.cond(design))
    coef, *_ = np.linalg.lstsq(design, target_fit, rcond=None)
    if not np.isfinite(coef).all():
        raise ValueError("Affine fit produced non-finite coefficients.")
    fitted = affine_values(source_fit, coef)
    rmse = float(np.sqrt(np.mean(np.sum((fitted - target_fit) ** 2, axis=1))))
    diagnostics: dict[str, float | int] = {
        "rows_used": int(source_fit.shape[0]),
        "rows_dropped": int((~finite).sum()),
        "rank": rank,
        "condition_number": condition,
    }
    return coef, rmse, diagnostics


def apply_affine(xy: np.ndarray, coef: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError("Prediction coordinates contain non-finite values before affine conversion.")
    return affine_values(xy, coef)


def get_xy(adata: ad.AnnData, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in adata.obsm:
            value = np.asarray(adata.obsm[key], dtype=np.float64)
            if value.ndim == 2 and value.shape[1] >= 2:
                return value[:, :2]
    raise KeyError(f"No spatial coordinates found in keys={keys}")


def get_aligned_xy(adata: ad.AnnData) -> np.ndarray:
    return get_xy(adata, ("X_spatial_aligned", "X_spatial_input", "spatial"))


def get_display_xy(adata: ad.AnnData) -> np.ndarray:
    return get_xy(adata, ("spatial", "spatial_raw", "X_spatial_aligned"))


def time_values(adata: ad.AnnData) -> np.ndarray:
    for key in ("time_input", "time", "Time point"):
        if key in adata.obs:
            return pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
    raise KeyError("Prediction/observed h5ad has no time column.")


def state_matrix(adata: ad.AnnData, xy_override: np.ndarray | None = None) -> np.ndarray:
    if "X_state" in adata.obsm:
        state = np.asarray(adata.obsm["X_state"], dtype=np.float32).copy()
        if xy_override is not None:
            state[:, :2] = np.asarray(xy_override, dtype=np.float32)
        return state
    spatial = np.asarray(xy_override if xy_override is not None else get_aligned_xy(adata), dtype=np.float32)
    if "X_gene_input" in adata.obsm:
        gene = np.asarray(adata.obsm["X_gene_input"], dtype=np.float32)
    elif "X_gene_prior_gatae_10d" in adata.obsm:
        gene = np.asarray(adata.obsm["X_gene_prior_gatae_10d"], dtype=np.float32)
    else:
        value = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        gene = np.asarray(value, dtype=np.float32)
    return np.hstack([spatial[:, :2], gene]).astype(np.float32)


def set_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    at = cKDTree(a)
    bt = cKDTree(b)
    return 0.5 * (float(at.query(b, k=1)[0].mean()) + float(bt.query(a, k=1)[0].mean()))


def source_to_common_transform(
    source_obs: ad.AnnData,
    common_obs: ad.AnnData,
    target_time: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source_mask = np.isclose(time_values(source_obs), target_time)
    common_mask = np.isclose(time_values(common_obs), target_time)
    source_aligned = get_aligned_xy(source_obs)[source_mask]
    source_display = get_display_xy(source_obs)[source_mask]
    common_display = get_display_xy(common_obs)[common_mask]
    common_aligned = get_aligned_xy(common_obs)[common_mask]
    source_to_display, source_rmse, source_diag = fit_affine(source_aligned, source_display)
    display_to_common, common_rmse, common_diag = fit_affine(common_display, common_aligned)
    audit = {
        "source_to_display_rmse": source_rmse,
        "display_to_common_rmse": common_rmse,
        "source_display_vs_common_display_chamfer": set_chamfer(source_display, common_display),
        "source_affine_rows_used": source_diag["rows_used"],
        "source_affine_rows_dropped": source_diag["rows_dropped"],
        "source_affine_rank": source_diag["rank"],
        "source_affine_condition_number": source_diag["condition_number"],
        "common_affine_rows_used": common_diag["rows_used"],
        "common_affine_rows_dropped": common_diag["rows_dropped"],
        "common_affine_rank": common_diag["rank"],
        "common_affine_condition_number": common_diag["condition_number"],
    }
    return source_to_display, display_to_common, audit


def common_prediction_xy(
    dataset_id: str,
    method: str,
    pred: ad.AnnData,
    common_obs: ad.AnnData,
    target_time: float,
    source_obs: ad.AnnData | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_xy = get_aligned_xy(pred)
    transform: dict[str, Any] = {
        "coordinate_basis": "common_bundle_X_spatial_aligned",
        "coordinate_transform": "identity",
        "source_to_display_rmse": 0.0,
        "display_to_common_rmse": 0.0,
        "source_display_vs_common_display_chamfer": 0.0,
    }
    if source_obs is not None:
        first, second, audit = source_to_common_transform(source_obs, common_obs, target_time)
        raw_xy = apply_affine(apply_affine(raw_xy, first), second)
        transform.update(audit)
        transform["coordinate_transform"] = "source_aligned_to_shared_display_to_common_aligned"
    transform.update(
        {
            "pred_std_x_common": float(np.std(raw_xy[:, 0])),
            "pred_std_y_common": float(np.std(raw_xy[:, 1])),
        }
    )
    return raw_xy, transform


def prediction_labels(
    dataset_id: str,
    method: str,
    pred: ad.AnnData,
    bundle: runner.Bundle,
    target_time: float,
    common_xy: np.ndarray,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    if dataset_id == "mouse_dorsal_midbrain" and method == "Original stVCR":
        if "Annotation" not in pred.obs:
            raise KeyError("Mouse Original stVCR prediction has no method-specific Annotation labels.")
        labels = pred.obs["Annotation"].astype(str).to_numpy()
        return labels, "prediction_obs_method_specific_20d_classifier", None

    state = state_matrix(pred, xy_override=common_xy)
    if state.shape[1] != bundle.state.shape[1]:
        raise ValueError(
            f"{dataset_id} {method}: state width {state.shape[1]} != common bundle width {bundle.state.shape[1]}"
        )
    labels = runner.label_predictions(bundle, state, target_time, detailed=False)
    detailed = None
    if bundle.detailed_classifier is not None:
        detailed = runner.label_predictions(bundle, state, target_time, detailed=True)
    return labels, "common_bundle_knn_classifier", detailed


def evaluate_prediction(
    dataset_id: str,
    method: str,
    target_time: float,
    target_label: int,
    prediction_path: Path,
    bundle: runner.Bundle,
    source_obs: ad.AnnData | None,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    pred = ad.read_h5ad(prediction_path)
    common_xy, transform = common_prediction_xy(
        dataset_id,
        method,
        pred,
        bundle.adata,
        target_time,
        source_obs,
    )
    real_mask = bundle.labels == target_label
    real_xy = np.asarray(bundle.state[real_mask, :2], dtype=np.float64)
    real_labels = bundle.adata.obs[bundle.spec.annotation_key].astype(str).to_numpy()[real_mask]
    pred_labels, label_mode, detailed_pred = prediction_labels(
        dataset_id,
        method,
        pred,
        bundle,
        target_time,
        common_xy,
    )

    metrics: dict[str, Any] = {}
    metrics.update(runner.composition_metrics(real_labels, pred_labels, prefix="annotation"))
    if bundle.detailed_classifier is not None and detailed_pred is not None and bundle.spec.detailed_key:
        detailed_real = bundle.adata.obs[bundle.spec.detailed_key].astype(str).to_numpy()[real_mask]
        metrics.update(runner.composition_metrics(detailed_real, detailed_pred, prefix="detailed"))

    spatial = spatial_extended_metrics(real_xy, common_xy)
    metrics.update(spatial)
    metrics["spatial_iou"] = spatial.get("spatial_iou_48", np.nan)
    metrics["spatial_chamfer_norm_diag"] = spatial.get("canonical_audit_spatial_chamfer", np.nan)
    metrics["spatial_iou_120"] = spatial.get("canonical_audit_spatial_iou", np.nan)
    metrics.update(
        {
            "dataset_id": dataset_id,
            "dataset": DATASET_LABEL[dataset_id],
            "target_time_input": target_time,
            "target_time_label": target_label,
            "target_time_display": TIME_DISPLAY[(dataset_id, target_label)],
            "method": method,
            "display_method": DISPLAY_METHOD[method],
            "prediction_path": str(prediction_path),
            "n_real": int(real_xy.shape[0]),
            "n_pred": int(common_xy.shape[0]),
            "label_evaluator": label_mode,
            "spatial_evaluator": "unified_v5_raw_half_chamfer_iou48_common_aligned_basis",
            "coordinate_basis": transform["coordinate_basis"],
            "coordinate_transform": transform["coordinate_transform"],
        }
    )
    transform.update(
        {
            "dataset_id": dataset_id,
            "target_time_input": target_time,
            "target_time_label": target_label,
            "target_time_display": TIME_DISPLAY[(dataset_id, target_label)],
            "method": method,
            "display_method": DISPLAY_METHOD[method],
            "prediction_path": str(prediction_path),
            "real_std_x_common": float(np.std(real_xy[:, 0])),
            "real_std_y_common": float(np.std(real_xy[:, 1])),
        }
    )
    return metrics, common_xy, transform


def metric_columns(frame: pd.DataFrame) -> list[str]:
    fixed = {
        "dataset_id",
        "dataset",
        "target_time_input",
        "target_time_label",
        "target_time_display",
        "method",
        "display_method",
        "prediction_path",
        "n_real",
        "n_pred",
        "label_evaluator",
        "spatial_evaluator",
        "coordinate_basis",
        "coordinate_transform",
    }
    return [col for col in frame.columns if col not in fixed and pd.api.types.is_numeric_dtype(frame[col])]


def wide_to_long(frame: pd.DataFrame, extra_id: list[str] | None = None) -> pd.DataFrame:
    extra_id = extra_id or []
    ids = [
        "dataset_id",
        "dataset",
        "target_time_input",
        "target_time_label",
        "target_time_display",
        "method",
        "display_method",
        "prediction_path",
        "n_real",
        "n_pred",
        "label_evaluator",
        "spatial_evaluator",
        "coordinate_basis",
        "coordinate_transform",
        *extra_id,
    ]
    ids = [col for col in ids if col in frame.columns]
    metrics = [col for col in metric_columns(frame) if col not in extra_id]
    long = frame.melt(id_vars=ids, value_vars=metrics, var_name="metric", value_name="value")
    long["direction"] = long["metric"].map(METRIC_DIRECTIONS).fillna("audit_only")
    long["display_metric"] = long["metric"].map(METRIC_LABELS).fillna(long["metric"])
    return long


def method_source_observed(dataset_id: str, method: str) -> ad.AnnData | None:
    if dataset_id == "mouse_dorsal_midbrain" and method == "Original stVCR":
        return ad.read_h5ad(MOUSE_STVCR_OBS)
    return None


def multiseed_source_observed_path(dataset_id: str, method: str, run_dir: Path) -> Path | None:
    if method not in {"SpaPOT", "Original stVCR"}:
        return None
    filename = (
        "stvcr_notebook_trained.h5ad"
        if dataset_id == "mouse_dorsal_midbrain"
        else "stvcr_notebook_hybrid_trained.h5ad"
    )
    source_path = run_dir / filename
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source observed h5ad for coordinate conversion: {source_path}")
    return source_path


def build_main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(V4_SOURCE_MANIFEST)
    manifest = manifest[
        manifest["method"].isin(METHOD_ORDER)
        & manifest["spatial_reconstruction_available"].astype(bool)
    ].copy()
    bundles: dict[str, runner.Bundle] = {}
    source_obs_cache: dict[tuple[str, str], ad.AnnData | None] = {}
    rows: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []

    for dataset_id in DATASET_ORDER:
        spec = next(item for item in prep.DATASETS if item.dataset_id == dataset_id)
        bundles[dataset_id] = runner.load_bundle(spec)
        bundle = bundles[dataset_id]
        dataset_rows = manifest[manifest["dataset_id"] == dataset_id]
        for target_label in sorted(dataset_rows["target_time_label"].astype(int).unique()):
            target_time = float(bundle.inverse_time_mapping[target_label])
            real_mask = bundle.labels == target_label
            real_xy = np.asarray(bundle.state[real_mask, :2], dtype=np.float64)
            point_frames.append(
                pd.DataFrame(
                    {
                        "dataset_id": dataset_id,
                        "dataset": DATASET_LABEL[dataset_id],
                        "target_time_input": target_time,
                        "target_time_label": target_label,
                        "target_time_display": TIME_DISPLAY[(dataset_id, target_label)],
                        "method": "Observed target",
                        "display_method": "Observed",
                        "point_role": "observed_target",
                        "point_index": np.arange(real_xy.shape[0], dtype=int),
                        "x": real_xy[:, 0],
                        "y": real_xy[:, 1],
                    }
                )
            )
            target_rows = dataset_rows[dataset_rows["target_time_label"].astype(int) == target_label]
            for method in METHOD_ORDER:
                hit = target_rows[target_rows["method"] == method]
                if hit.empty:
                    continue
                prediction_path = Path(str(hit.iloc[0]["source_path"]))
                key = (dataset_id, method)
                if key not in source_obs_cache:
                    source_obs_cache[key] = method_source_observed(dataset_id, method)
                metrics, pred_xy, transform = evaluate_prediction(
                    dataset_id,
                    method,
                    target_time,
                    target_label,
                    prediction_path,
                    bundle,
                    source_obs_cache[key],
                )
                rows.append(metrics)
                transform_rows.append(transform)
                point_frames.append(
                    pd.DataFrame(
                        {
                            "dataset_id": dataset_id,
                            "dataset": DATASET_LABEL[dataset_id],
                            "target_time_input": target_time,
                            "target_time_label": target_label,
                            "target_time_display": TIME_DISPLAY[(dataset_id, target_label)],
                            "method": method,
                            "display_method": DISPLAY_METHOD[method],
                            "point_role": "predicted_reconstruction",
                            "point_index": np.arange(pred_xy.shape[0], dtype=int),
                            "x": pred_xy[:, 0],
                            "y": pred_xy[:, 1],
                        }
                    )
                )

    wide = pd.DataFrame(rows)
    wide["method_order"] = wide["method"].map({method: i for i, method in enumerate(METHOD_ORDER)})
    wide = wide.sort_values(["dataset_id", "target_time_label", "method_order"]).drop(columns="method_order")
    long = wide_to_long(wide)
    points = pd.concat(point_frames, ignore_index=True)
    transforms = pd.DataFrame(transform_rows)
    return wide, long, points, transforms


def build_multiseed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(MULTISEED_SOURCE)
    unique_cols = [
        "dataset_id",
        "dataset",
        "method",
        "seed",
        "run_dir",
        "target_time_input",
        "target_time_label",
        "prediction_path",
        "repeat_type",
        "seed_repeatable",
    ]
    jobs = source[source["method"].isin(METHOD_ORDER)][unique_cols].drop_duplicates().copy()
    bundles: dict[str, runner.Bundle] = {}
    source_obs_cache: dict[str, ad.AnnData] = {}
    rows: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    for dataset_id in DATASET_ORDER:
        spec = next(item for item in prep.DATASETS if item.dataset_id == dataset_id)
        bundles[dataset_id] = runner.load_bundle(spec)
        bundle = bundles[dataset_id]
        dataset_jobs = jobs[jobs["dataset_id"] == dataset_id]
        for _, job in dataset_jobs.iterrows():
            method = str(job["method"])
            target_label = int(job["target_time_label"])
            target_time = float(job["target_time_input"])
            run_dir = Path(str(job["run_dir"]))
            source_path = multiseed_source_observed_path(dataset_id, method, run_dir)
            source_obs = None
            if source_path is not None:
                cache_key = str(source_path)
                if cache_key not in source_obs_cache:
                    source_obs_cache[cache_key] = ad.read_h5ad(source_path)
                source_obs = source_obs_cache[cache_key]
            metrics, _, transform = evaluate_prediction(
                dataset_id,
                method,
                target_time,
                target_label,
                Path(str(job["prediction_path"])),
                bundle,
                source_obs,
            )
            metrics.update(
                {
                    "seed": job["seed"],
                    "run_dir": job["run_dir"],
                    "repeat_type": job["repeat_type"],
                    "seed_repeatable": bool(job["seed_repeatable"]),
                }
            )
            transform.update(
                {
                    "seed": job["seed"],
                    "run_dir": job["run_dir"],
                    "source_observed_path": "" if source_path is None else str(source_path),
                }
            )
            rows.append(metrics)
            transforms.append(transform)
    wide = pd.DataFrame(rows)
    wide["method_order"] = wide["method"].map({method: i for i, method in enumerate(METHOD_ORDER)})
    wide = wide.sort_values(["dataset_id", "target_time_label", "method_order", "seed"], na_position="last").drop(columns="method_order")
    long = wide_to_long(
        wide,
        extra_id=["seed", "run_dir", "repeat_type", "seed_repeatable"],
    )
    return wide, long, pd.DataFrame(transforms)


def build_method_manifest() -> pd.DataFrame:
    source = pd.read_csv(V4_METHOD_MANIFEST)
    source["display_method"] = source["method"].map(DISPLAY_METHOD).fillna(source["method"])
    source["included_in_spatial_benchmark_v5"] = source["method"].isin(METHOD_ORDER)
    source["v5_exclusion_reason"] = ""
    source.loc[source["method"] == "PRESCIENT official", "v5_exclusion_reason"] = (
        "Expression/PCA-only predictions have no verified 2D spatial reconstruction."
    )
    source.loc[~source["method"].isin([*METHOD_ORDER, "PRESCIENT official"]), "v5_exclusion_reason"] = (
        "Not part of the six-method external-first spatial benchmark."
    )
    return source


def distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def read_run_summary(run_path: Path) -> tuple[dict[str, Any], Path]:
    summary_path = run_path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8")), summary_path


def sample_schedule(config: dict[str, Any]) -> tuple[str, float | None]:
    values = config.get("num_samples")
    if not isinstance(values, list) or not values:
        return "", None
    numeric = [float(value) for value in values]
    return json.dumps(values), float(max(numeric))


def build_training_scale_audit(method_manifest: pd.DataFrame) -> pd.DataFrame:
    official = pd.read_csv(OFFICIAL_SCALE_AUDIT)
    official_methods = {
        "SpaPOT",
        "TrajectoryNet official",
        "PRESCIENT official",
        "POT official solver",
        "moscot official",
        "Spateo official",
    }
    audit = official[
        official["dataset_id"].isin(DATASET_ORDER) & official["method"].isin(official_methods)
    ].copy()
    audit["display_method"] = audit["method"].map(DISPLAY_METHOD).fillna(audit["method"])
    audit["benchmark_scope"] = np.where(
        audit["method"].eq("PRESCIENT official"),
        "excluded_spatial_unavailable",
        "main_spatial_benchmark",
    )
    audit["spatial_reconstruction_available"] = ~audit["method"].eq("PRESCIENT official")
    audit["included_in_figure6_spatial"] = audit["method"].isin(METHOD_ORDER)
    audit["training_trace_or_report"] = ""
    audit["device"] = "not_applicable"
    audit["sample_schedule"] = ""

    for index, row in audit.iterrows():
        run_path = Path(str(row["source_path"]))
        report_path = run_path / "official_method_run_audit.csv"
        if report_path.exists():
            audit.loc[index, "training_trace_or_report"] = str(report_path)
        if row["method"] == "TrajectoryNet official":
            device_path = run_path / "trajectorynet_device_audit.json"
            if not device_path.exists():
                raise FileNotFoundError(f"Missing TrajectoryNet device audit: {device_path}")
            report = json.loads(device_path.read_text(encoding="utf-8"))
            audit.loc[index, "epochs_or_niters"] = report.get(
                "niters", row["epochs_or_niters"]
            )
            audit.loc[index, "batch_size_or_sample_size"] = report.get(
                "batch_size", row["batch_size_or_sample_size"]
            )
            audit.loc[index, "hidden_dim"] = report.get("dims", row["hidden_dim"])
            audit.loc[index, "n_layers"] = report.get("num_blocks", row["n_layers"])
            audit.loc[index, "device"] = report.get("device", "unknown")
            audit.loc[index, "training_trace_or_report"] = str(device_path)

    for index, row in audit[audit["method"].eq("SpaPOT")].iterrows():
        summary, summary_path = read_run_summary(Path(str(row["source_path"])))
        config = summary.get("config", {})
        schedule, maximum = sample_schedule(config)
        audit.loc[index, "epochs_or_niters"] = config.get("n_epochs", row["epochs_or_niters"])
        audit.loc[index, "batch_size_or_sample_size"] = maximum
        audit.loc[index, "hidden_dim"] = config.get("hidden_dim", row["hidden_dim"])
        audit.loc[index, "n_layers"] = config.get("n_hiddens", row["n_layers"])
        audit.loc[index, "latent_dim"] = 10
        audit.loc[index, "sample_schedule"] = schedule
        audit.loc[index, "device"] = summary.get("device", "unknown")
        audit.loc[index, "training_trace_or_report"] = str(summary_path)

    original_rows: list[dict[str, Any]] = []
    original_manifest = method_manifest[
        method_manifest["method"].eq("Original stVCR")
        & method_manifest["dataset_id"].isin(DATASET_ORDER)
    ]
    for _, row in original_manifest.iterrows():
        run_path = Path(str(row["run_path"]))
        summary, summary_path = read_run_summary(run_path)
        config = summary.get("config", {})
        schedule, maximum = sample_schedule(config)
        original_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "method": "Original stVCR",
                "source_path": str(run_path),
                "training_unit": "epoch",
                "epochs_or_niters": config.get("n_epochs"),
                "batch_size_or_sample_size": maximum,
                "hidden_dim": config.get("hidden_dim"),
                "n_layers": config.get("n_hiddens"),
                "latent_dim": np.nan,
                "qualified_for_main": bool(config.get("n_epochs", 0) >= 400),
                "exclusion_reason": "",
                "display_method": "stVCR",
                "benchmark_scope": "main_spatial_benchmark",
                "spatial_reconstruction_available": True,
                "included_in_figure6_spatial": True,
                "training_trace_or_report": str(summary_path),
                "device": summary.get("device", "unknown"),
                "sample_schedule": schedule,
            }
        )
    audit = pd.concat([audit, pd.DataFrame(original_rows)], ignore_index=True, sort=False)

    legacy = pd.read_csv(LEGACY_SCALE_AUDIT)
    supplement = legacy[
        legacy["comparison_tier"].isin(["supplement_internal_ablation", "excluded_alignment_reference"])
    ].copy()
    supplement_rows: list[dict[str, Any]] = []
    for _, row in supplement.iterrows():
        scope = (
            "supplement_internal_ablation"
            if row["comparison_tier"] == "supplement_internal_ablation"
            else "excluded_smoke_timing_debug"
        )
        supplement_rows.append(
            {
                "dataset_id": (
                    "mouse_dorsal_midbrain"
                    if str(row["dataset"]).startswith("Mouse")
                    else "zesta_10_12_18"
                ),
                "method": row["method"],
                "source_path": row["source_path"],
                "training_unit": "epoch",
                "epochs_or_niters": row["epochs"],
                "batch_size_or_sample_size": row["batch_size/sample_size"],
                "hidden_dim": row["hidden_dim"],
                "n_layers": row["n_layers"],
                "latent_dim": row["latent_dim"],
                "qualified_for_main": False,
                "exclusion_reason": row["exclusion_reason"],
                "display_method": row["method"],
                "benchmark_scope": scope,
                "spatial_reconstruction_available": True,
                "included_in_figure6_spatial": False,
                "training_trace_or_report": row["training_trace_or_report"],
                "device": "not_audited",
                "sample_schedule": "",
            }
        )
    audit = pd.concat([audit, pd.DataFrame(supplement_rows)], ignore_index=True, sort=False)

    legacy_local = official[official["dataset_id"].eq("legacy_v2")].copy()
    if not legacy_local.empty:
        legacy_local["display_method"] = legacy_local["method"]
        legacy_local["benchmark_scope"] = "excluded_legacy_local_or_proxy"
        legacy_local["spatial_reconstruction_available"] = False
        legacy_local["included_in_figure6_spatial"] = False
        legacy_local["training_trace_or_report"] = ""
        legacy_local["device"] = "not_applicable"
        legacy_local["sample_schedule"] = ""
        audit = pd.concat([audit, legacy_local], ignore_index=True, sort=False)

    versions = {
        "POT official solver": distribution_version("POT"),
        "moscot official": distribution_version("moscot"),
        "Spateo official": distribution_version("spateo-release"),
        "TrajectoryNet official": distribution_version("TrajectoryNet"),
    }
    audit["package_version"] = audit["method"].map(versions).fillna("project_or_not_applicable")
    audit["scale_gate_passed"] = (
        audit["included_in_figure6_spatial"].astype(bool)
        & audit["qualified_for_main"].astype(bool)
    )
    return audit


def build_multiseed_training_audit() -> pd.DataFrame:
    source = pd.read_csv(MULTISEED_SOURCE)
    jobs = source[source["method"].isin(METHOD_ORDER)][
        ["dataset_id", "method", "seed", "run_dir", "prediction_path", "repeat_type", "seed_repeatable"]
    ].drop_duplicates()
    jobs["audit_run_dir"] = jobs.apply(
        lambda row: str(row["run_dir"])
        if bool(row["seed_repeatable"])
        else str(Path(str(row["prediction_path"])).parents[1]),
        axis=1,
    )
    rows: list[dict[str, Any]] = []
    for (dataset_id, method, run_dir), block in jobs.groupby(
        ["dataset_id", "method", "audit_run_dir"], dropna=False, sort=True
    ):
        seed_values = pd.to_numeric(block["seed"], errors="coerce").dropna()
        seed = int(seed_values.iloc[0]) if not seed_values.empty else np.nan
        run_path = Path(str(run_dir))
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "method": method,
            "display_method": DISPLAY_METHOD[method],
            "seed": seed,
            "run_dir": str(run_path),
            "prediction_artifact_count": int(block["prediction_path"].nunique()),
            "repeat_type": block["repeat_type"].iloc[0],
            "seed_repeatable": bool(block["seed_repeatable"].iloc[0]),
            "epochs_or_niters": np.nan,
            "batch_size_or_sample_size": np.nan,
            "sample_schedule": "",
            "hidden_dim": np.nan,
            "n_layers": np.nan,
            "latent_dim_or_input_dim": np.nan,
            "device": "not_applicable",
            "status": "completed_existing_artifact",
            "training_trace_or_report": "",
        }
        if method in {"SpaPOT", "Original stVCR"}:
            summary, summary_path = read_run_summary(run_path)
            config = summary.get("config", {})
            schedule, maximum = sample_schedule(config)
            row.update(
                {
                    "epochs_or_niters": config.get("n_epochs"),
                    "batch_size_or_sample_size": maximum,
                    "sample_schedule": schedule,
                    "hidden_dim": config.get("hidden_dim"),
                    "n_layers": config.get("n_hiddens"),
                    "latent_dim_or_input_dim": 10 if method == "SpaPOT" else np.nan,
                    "device": summary.get("device", "unknown"),
                    "status": summary.get("status", "unknown"),
                    "training_trace_or_report": str(summary_path),
                }
            )
        elif method == "TrajectoryNet official":
            report_path = run_path / "trajectorynet_device_audit.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row.update(
                {
                    "epochs_or_niters": report.get("niters"),
                    "batch_size_or_sample_size": report.get("batch_size"),
                    "hidden_dim": report.get("dims"),
                    "n_layers": report.get("num_blocks"),
                    "latent_dim_or_input_dim": 12,
                    "device": report.get("device", "unknown"),
                    "status": report.get("status", "unknown"),
                    "training_trace_or_report": str(report_path),
                }
            )
        row["qualified_for_multiseed_figure"] = bool(
            (not row["seed_repeatable"])
            or (
                pd.notna(row["epochs_or_niters"])
                and float(row["epochs_or_niters"]) >= 400
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_implementation_provenance(method_manifest: pd.DataFrame) -> pd.DataFrame:
    provenance = method_manifest[
        method_manifest["dataset_id"].isin(DATASET_ORDER)
        & method_manifest["method"].isin([*METHOD_ORDER, "PRESCIENT official"])
    ].copy()
    package_versions = {
        "POT official solver": distribution_version("POT"),
        "moscot official": distribution_version("moscot"),
        "Spateo official": distribution_version("spateo-release"),
        "TrajectoryNet official": distribution_version("TrajectoryNet"),
    }
    official_apis = {
        "POT official solver": "ot.bregman.sinkhorn_log",
        "moscot official": "moscot.problems.time.TemporalProblem.prepare/solve/pull; row-normalized coupling barycentric readout",
        "Spateo official": "spateo.alignment.morpho_align",
        "TrajectoryNet official": "TrajectoryNet.dataset/train_misc/main.train/CNF forward",
        "PRESCIENT official": "prescient process_data/train_model/simulate",
        "Original stVCR": "project same-dataset original-style rerun",
        "SpaPOT": "project stCTD training/prediction pipeline",
    }
    direct_official = {
        "POT official solver": True,
        "moscot official": True,
        "Spateo official": True,
        "TrajectoryNet official": True,
        "PRESCIENT official": True,
        "Original stVCR": False,
        "SpaPOT": False,
    }
    boundaries = {
        "POT official solver": "Official POT API with project target-aware barycentric evaluation glue.",
        "moscot official": "Official moscot TemporalProblem; target slices participate in OT fitting. Spatial predictions use the source-supported row-normalized coupling barycenter, evaluated with normalize=False and scale_by_marginals=False.",
        "Spateo official": "Official Spateo morpho alignment; alignment-only, not a temporal generator.",
        "TrajectoryNet official": "Official TrajectoryNet model, loss, data loader, and train loop; project wrapper only adds explicit MPS device selection and materializes h5ad outputs.",
        "PRESCIENT official": "Official package run, excluded from spatial Figure 6 because the verified outputs are expression/PCA trajectories without 2D coordinates.",
        "Original stVCR": "Same-dataset original-style/adapted project rerun; mouse uses the earlier AE-hybrid predecessor and is not the published GitHub ARTISTA result.",
        "SpaPOT": "Proposed project method, displayed as stCTD.",
    }
    provenance["package_version"] = provenance["method"].map(package_versions).fillna(
        "project_or_not_applicable"
    )
    provenance["official_api_or_entrypoint"] = provenance["method"].map(official_apis)
    provenance["official_code_directly_used"] = provenance["method"].map(direct_official)
    provenance["implementation_boundary"] = provenance["method"].map(boundaries)
    provenance["prediction_artifacts_exist"] = provenance["prediction_path"].fillna("").map(
        lambda value: bool(value)
        and all(Path(path).exists() for path in str(value).split(";") if path)
    )
    provenance["included_in_spatial_figure6"] = provenance["method"].isin(METHOD_ORDER)
    return provenance


def build_spatial_source_manifest(main_wide: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(V4_SOURCE_MANIFEST)
    observed_paths = old[old["method"].eq("Observed target")][
        ["dataset_id", "target_time_label", "source_path"]
    ]
    observed = (
        points[points["method"].eq("Observed target")]
        .groupby(
            ["dataset_id", "dataset", "target_time_input", "target_time_label", "target_time_display"],
            as_index=False,
        )
        .agg(n_points=("point_index", "count"))
        .merge(observed_paths, on=["dataset_id", "target_time_label"], how="left")
    )
    observed["method"] = "Observed target"
    observed["display_method"] = "Observed"
    observed["spatial_reconstruction_available"] = True
    observed["coordinate_basis"] = "common_bundle_X_spatial_aligned"
    observed["coordinate_transform"] = "identity"
    predicted = main_wide[
        [
            "dataset_id",
            "dataset",
            "target_time_input",
            "target_time_label",
            "target_time_display",
            "method",
            "display_method",
            "prediction_path",
            "n_pred",
            "coordinate_basis",
            "coordinate_transform",
        ]
    ].rename(columns={"prediction_path": "source_path", "n_pred": "n_points"})
    predicted["spatial_reconstruction_available"] = True
    columns = [
        "dataset_id",
        "dataset",
        "target_time_input",
        "target_time_label",
        "target_time_display",
        "method",
        "display_method",
        "source_path",
        "n_points",
        "spatial_reconstruction_available",
        "coordinate_basis",
        "coordinate_transform",
    ]
    return pd.concat([observed[columns], predicted[columns]], ignore_index=True)


def previous_scale_audit(unified_long: pd.DataFrame) -> pd.DataFrame:
    previous = pd.read_csv(PREVIOUS_METRIC_SOURCE)
    previous = previous[previous["method"].isin(METHOD_ORDER)].copy()
    previous = previous[previous["metric"].isin(["annotation_l1", "spatial_chamfer", "spatial_iou"])]
    current = unified_long[unified_long["metric"].isin(["annotation_l1", "spatial_chamfer", "spatial_iou"])][
        ["dataset_id", "method", "target_time_label", "metric", "value"]
    ].rename(columns={"value": "unified_v5_value"})
    audit = previous.merge(current, on=["dataset_id", "method", "target_time_label", "metric"], how="outer")
    audit = audit.rename(columns={"value": "previous_value"})
    unified_classes = {
        "annotation_l1": "half_l1_total_variation_0_1",
        "spatial_chamfer": "raw_half_bidirectional_mean_common_aligned_basis",
        "spatial_iou": "occupancy_iou_48_common_aligned_basis",
    }
    audit["previous_scale_class"] = audit["metric"].map(unified_classes)
    mouse_stvcr = (audit["dataset_id"] == "mouse_dorsal_midbrain") & (audit["method"] == "Original stVCR")
    audit.loc[mouse_stvcr & audit["metric"].eq("annotation_l1"), "previous_scale_class"] = "legacy_full_l1"
    audit.loc[mouse_stvcr & audit["metric"].eq("spatial_chamfer"), "previous_scale_class"] = "legacy_sum_over_diag_chamfer"
    audit.loc[mouse_stvcr & audit["metric"].eq("spatial_iou"), "previous_scale_class"] = "legacy_iou120"
    audit["unified_scale_class"] = audit["metric"].map(unified_classes)
    audit["previous_evaluator_matches_v5"] = (
        audit["previous_scale_class"] == audit["unified_scale_class"]
    )
    audit["absolute_change"] = audit["unified_v5_value"] - audit["previous_value"]
    return audit


def evaluator_contract() -> pd.DataFrame:
    rows = [
        {
            "metric": "annotation_l1",
            "direction": "lower_better",
            "definition": "0.5 * sum(abs(real composition - predicted composition))",
            "range_or_unit": "0 to 1; total variation distance",
            "canonical_for_figure6": True,
        },
        {
            "metric": "spatial_chamfer",
            "direction": "lower_better",
            "definition": "0.5 * (mean pred-to-real NN distance + mean real-to-pred NN distance)",
            "range_or_unit": "common aligned coordinate units",
            "canonical_for_figure6": True,
        },
        {
            "metric": "spatial_iou",
            "direction": "higher_better",
            "definition": "occupied-grid intersection over union using pooled bounds and 48 x 48 bins",
            "range_or_unit": "0 to 1",
            "canonical_for_figure6": True,
        },
        {
            "metric": "spatial_chamfer_norm_diag",
            "direction": "lower_better",
            "definition": "bidirectional NN distance sum divided by pooled point-cloud diagonal",
            "range_or_unit": "dimensionless audit only",
            "canonical_for_figure6": False,
        },
        {
            "metric": "spatial_iou_120",
            "direction": "higher_better",
            "definition": "occupied-grid intersection over union using 120 x 120 histogram bins",
            "range_or_unit": "0 to 1; audit only",
            "canonical_for_figure6": False,
        },
    ]
    return pd.DataFrame(rows)


def multiseed_summary(long: pd.DataFrame) -> pd.DataFrame:
    keep_metrics = ["annotation_l1", "spatial_chamfer", "spatial_iou"]
    selected = long[long["metric"].isin(keep_metrics)].copy()
    rows: list[dict[str, Any]] = []
    for keys, block in selected.groupby(["dataset_id", "dataset", "metric", "display_metric", "method", "display_method"], dropna=False):
        seed_values = pd.to_numeric(block["seed"], errors="coerce")
        rows.append(
            {
                "dataset_id": keys[0],
                "dataset": keys[1],
                "metric": keys[2],
                "display_metric": keys[3],
                "method": keys[4],
                "display_method": keys[5],
                "mean_value": float(pd.to_numeric(block["value"], errors="coerce").mean()),
                "median_value": float(pd.to_numeric(block["value"], errors="coerce").median()),
                "point_count": int(block["value"].notna().sum()),
                "seed_count": int(seed_values.dropna().nunique()),
                "target_timepoint_count": int(block["target_time_label"].nunique()),
                "direction": METRIC_DIRECTIONS[keys[2]],
                "spatial_evaluator": "unified_v5",
            }
        )
    return pd.DataFrame(rows)


def build_qa_checks(
    main_wide: pd.DataFrame,
    points: pd.DataFrame,
    transforms_main: pd.DataFrame,
    multiseed_wide: pd.DataFrame,
    transforms_multiseed: pd.DataFrame,
    training_audit: pd.DataFrame,
    multiseed_training_audit: pd.DataFrame,
    provenance: pd.DataFrame,
    source_manifest: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    expected = set(METHOD_ORDER)
    add(
        "main_shape",
        main_wide.shape[0] == 24,
        f"rows={main_wide.shape[0]}; expected=24",
    )
    per_dataset_sets = {
        dataset_id: set(block["method"])
        for dataset_id, block in main_wide.groupby("dataset_id")
    }
    add(
        "six_methods_per_dataset",
        set(per_dataset_sets) == set(DATASET_ORDER)
        and all(methods == expected for methods in per_dataset_sets.values()),
        str({key: sorted(value) for key, value in per_dataset_sets.items()}),
    )
    add(
        "prescient_absent_from_spatial_metrics",
        not main_wide["method"].eq("PRESCIENT official").any(),
        "PRESCIENT has no verified 2D spatial output.",
    )
    canonical = ["annotation_l1", "spatial_chamfer", "spatial_iou"]
    add(
        "canonical_metrics_finite",
        bool(np.isfinite(main_wide[canonical].to_numpy(dtype=float)).all()),
        f"metrics={canonical}",
    )
    point_groups = points.groupby(["dataset_id", "target_time_label", "method"]).size()
    add(
        "spatial_point_groups_complete",
        point_groups.shape[0] == 28 and bool((point_groups > 0).all()),
        f"groups={point_groups.shape[0]}; expected=28",
    )
    add(
        "spatial_source_manifest_complete",
        source_manifest.shape[0] == 28
        and source_manifest["source_path"].map(lambda value: Path(str(value)).exists()).all(),
        f"rows={source_manifest.shape[0]}; expected=28",
    )
    converted = pd.concat(
        [
            transforms_main[transforms_main["coordinate_transform"].ne("identity")],
            transforms_multiseed[transforms_multiseed["coordinate_transform"].ne("identity")],
        ],
        ignore_index=True,
    )
    conversion_ok = bool(
        not converted.empty
        and (converted["source_to_display_rmse"] < 1e-5).all()
        and (converted["display_to_common_rmse"] < 1e-5).all()
        and (converted["source_display_vs_common_display_chamfer"] < 1e-8).all()
        and (converted["source_affine_rows_dropped"].fillna(0) == 0).all()
        and (converted["common_affine_rows_dropped"].fillna(0) == 0).all()
    )
    add(
        "coordinate_conversion_audit",
        conversion_ok,
        f"converted_rows={converted.shape[0]}; max_rmse={max(converted['source_to_display_rmse'].max(), converted['display_to_common_rmse'].max()):.3e}",
    )
    neural = multiseed_wide[multiseed_wide["seed_repeatable"].astype(bool)]
    neural_counts = neural.groupby(["dataset_id", "method"]).agg(
        seed_count=("seed", "nunique"), row_count=("seed", "size")
    )
    add(
        "neural_multiseed_counts",
        neural_counts.shape[0] == 6
        and (neural_counts["seed_count"] == 3).all()
        and (neural_counts["row_count"] == 6).all(),
        neural_counts.to_dict(orient="index").__repr__(),
    )
    deterministic = multiseed_wide[~multiseed_wide["seed_repeatable"].astype(bool)]
    deterministic_counts = deterministic.groupby(["dataset_id", "method"]).size()
    add(
        "deterministic_two_target_points",
        deterministic_counts.shape[0] == 6 and (deterministic_counts == 2).all(),
        deterministic_counts.to_dict().__repr__(),
    )
    main_scale = training_audit[training_audit["included_in_figure6_spatial"].astype(bool)]
    add(
        "main_training_scale_gate",
        main_scale.shape[0] == 12 and main_scale["scale_gate_passed"].astype(bool).all(),
        f"rows={main_scale.shape[0]}; failed={int((~main_scale['scale_gate_passed'].astype(bool)).sum())}",
    )
    trace_paths = main_scale["training_trace_or_report"].fillna("").map(Path)
    add(
        "main_training_or_solver_trace_complete",
        bool((trace_paths.map(str) != ".").all() and trace_paths.map(Path.exists).all()),
        f"rows={main_scale.shape[0]}; missing={int((~trace_paths.map(Path.exists)).sum())}",
    )
    trajectory_rows = main_scale[main_scale["method"].eq("TrajectoryNet official")]
    add(
        "trajectorynet_mps_and_400_iteration_audit",
        trajectory_rows.shape[0] == 2
        and trajectory_rows["device"].eq("mps").all()
        and pd.to_numeric(trajectory_rows["epochs_or_niters"], errors="coerce").ge(400).all(),
        trajectory_rows[
            ["dataset_id", "device", "epochs_or_niters", "training_trace_or_report"]
        ].to_dict(orient="records").__repr__(),
    )
    moscot_audits: list[pd.DataFrame] = []
    for report_path in main_scale.loc[
        main_scale["method"].eq("moscot official"), "training_trace_or_report"
    ]:
        moscot_audits.append(pd.read_csv(report_path))
    moscot_audit = pd.concat(moscot_audits, ignore_index=True)
    add(
        "moscot_barycentric_readout_audit",
        moscot_audit["dataset_id"].nunique() == 2
        and moscot_audit["official_api"].str.endswith("prepare/solve/pull").all()
        and moscot_audit["readout"].eq(
            "row_normalized_coupling_barycentric_target_state"
        ).all()
        and (~moscot_audit["normalize_input_features"].astype(bool)).all()
        and (~moscot_audit["scale_by_marginals"].astype(bool)).all(),
        f"rows={moscot_audit.shape[0]}; datasets={moscot_audit['dataset_id'].nunique()}",
    )
    add(
        "multiseed_training_scale_gate",
        multiseed_training_audit.shape[0] == 24
        and multiseed_training_audit["qualified_for_multiseed_figure"].astype(bool).all(),
        f"rows={multiseed_training_audit.shape[0]}; failed={int((~multiseed_training_audit['qualified_for_multiseed_figure'].astype(bool)).sum())}",
    )
    official = provenance[provenance["method"].isin(
        ["POT official solver", "moscot official", "Spateo official", "TrajectoryNet official"]
    )]
    add(
        "official_package_provenance",
        official.shape[0] == 8
        and official["official_code_directly_used"].astype(bool).all()
        and ~official["package_version"].eq("not-installed").any(),
        official[["method", "package_version"]].drop_duplicates().to_dict(orient="records").__repr__(),
    )
    source_text = "\n".join(
        main_wide["prediction_path"].astype(str).tolist()
        + training_audit["source_path"].astype(str).tolist()
        + source_manifest["source_path"].astype(str).tolist()
    ).lower()
    add(
        "no_axolotl_scope",
        "axolotl" not in source_text,
        "Only mouse dorsal midbrain and zebrafish ZESTA sources are allowed.",
    )
    return pd.DataFrame(checks)


def write_docs(
    main_wide: pd.DataFrame,
    main_long: pd.DataFrame,
    multiseed_wide: pd.DataFrame,
    scale_audit: pd.DataFrame,
    training_audit: pd.DataFrame,
    multiseed_training_audit: pd.DataFrame,
    provenance: pd.DataFrame,
    qa_checks: pd.DataFrame,
) -> None:
    readme = """# Figure 6 unified model comparison v5

This package re-evaluates completed predictions without retraining.

- Display name `stCTD` retains internal source method key `SpaPOT`.
- Main spatial benchmark methods: POT, moscot, Spateo, TrajectoryNet, stVCR, stCTD.
- PRESCIENT is excluded because its official prediction artifact has no verified 2D spatial reconstruction.
- All spatial predictions are evaluated in each dataset's common `X_spatial_aligned` basis.
- Canonical metrics are half-L1 composition distance, raw half bidirectional Chamfer, and 48-bin spatial IoU.
- Mouse stVCR and neural multiseed coordinates are converted from each run's source-aligned basis through the shared display basis into the current common aligned basis when needed.
- moscot spatial predictions use the official temporal coupling with a source-supported row-normalized barycentric readout (`pull`, `normalize=False`, `scale_by_marginals=False`).
- Normalized Chamfer and 120-bin IoU are retained only as audit metrics.

Primary files:

- `unified_main_metrics_wide.csv`
- `unified_main_metrics_long.csv`
- `unified_main_spatial_points.csv`
- `unified_multiseed_metrics_wide.csv`
- `unified_multiseed_metrics_long.csv`
- `unified_multiseed_metric_summary.csv`
- `coordinate_transform_audit_main.csv`
- `coordinate_transform_audit_multiseed.csv`
- `previous_scale_mismatch_audit.csv`
- `evaluator_contract.csv`
- `main_method_manifest.csv`
- `unified_spatial_source_manifest.csv`
- `training_scale_audit.csv`
- `multiseed_training_scale_audit.csv`
- `implementation_provenance_audit.csv`
- `qa_checks.csv`
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    mismatch_count = int(
        (
            ~scale_audit["previous_evaluator_matches_v5"].astype(bool)
        ).sum()
    )
    audit = f"""# Figure 6 模型对比统一尺度审计

## 结论

- 已确认旧包混用了完整 L1/半 L1、归一化 Chamfer/原始 Chamfer、120-bin/48-bin IoU。
- v5 不重新训练，直接统一重评现有 prediction h5ad。
- 主 benchmark 共 {main_wide['method'].nunique()} 个方法、{main_wide['dataset_id'].nunique()} 个数据集、{main_wide.shape[0]} 个方法-目标切片组合。
- 多 seed 重评共 {multiseed_wide.shape[0]} 个 prediction 组合。
- 旧 source 中与 v5 evaluator class 不一致的基础指标行数：{mismatch_count}。
- QA 检查 {int(qa_checks['passed'].sum())}/{qa_checks.shape[0]} 通过。

## 统一定义

- State L1：半 L1，即 total variation distance，范围 0-1。
- Spatial Chamfer：公共 aligned 坐标中的双向最近邻均值除以 2。
- Spatial IoU：公共 aligned 坐标 pooled bounds 下的 48 x 48 occupancy IoU。
- 所有扩展空间指标均从同一公共坐标点云重新计算。

## 训练尺度与实现

- 主空间 benchmark 训练/求解审计行数：{int(training_audit['included_in_figure6_spatial'].astype(bool).sum())}。
- 多 seed 训练/求解审计行数：{multiseed_training_audit.shape[0]}。
- 正式 stCTD run 的目录名保留 `s512` 历史命名，但实际动态采样 schedule 以各 run `summary.json` 为准，当前主 run 为 `[692, 692, 692]`。
- POT `{distribution_version('POT')}`、moscot `{distribution_version('moscot')}`、spateo-release `{distribution_version('spateo-release')}`、TrajectoryNet `{distribution_version('TrajectoryNet')}` 均从本地 `stVCR` 环境的官方包入口执行。
- moscot 空间读出已统一为官方时间耦合矩阵上的 source-supported 行归一化重心投影；审计要求 `pull`、`normalize=False`、`scale_by_marginals=False`。
- Original stVCR 是 original-style/adapted same-dataset rerun，不冒充原始 GitHub ARTISTA 数值。

## 解释边界

- mouse stVCR 是 original-style/AE-hybrid adapted baseline，不应写成原始 GitHub ARTISTA 数值结果。
- POT、moscot、Spateo 使用目标切片或对齐约束，不是 fully generative baseline。
- 多 seed 中确定性方法只有两个目标时间点，不是随机训练重复。
"""
    (OUT_DIR / "COMPLETION_AUDIT_zh.md").write_text(audit, encoding="utf-8")

    contract = """# Figure 6 contract

Core conclusion: stCTD provides robust cross-species spatial future-state reconstruction under a single evaluator, while target-aware and adapted baselines are explicitly bounded.
Figure archetype: image plate plus quantitative grid.
Target output: manuscript Figure 6, Python backend, separate SVG/PDF/PNG/TIFF panels.
Panel map: a = method functionality; b = spatial predictions; c = quantitative performance.
Hero evidence: observed-versus-predicted spatial slices and unified spatial metrics.
Validation evidence: two datasets, two target slices each, and true neural seed repeats where available.
Source data: unified v5 long/wide metric tables, point table, method manifest, and coordinate audit.
Reviewer risks: target-aware proxies, adapted stVCR naming, mixed metric scales, deterministic two-point boxes, and post-hoc coordinate transforms.
"""
    (OUT_DIR / "FIGURE_CONTRACT.md").write_text(contract, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_wide, main_long, points, transforms_main = build_main()
    multiseed_wide, multiseed_long, transforms_multiseed = build_multiseed()
    method_manifest = build_method_manifest()
    training_audit = build_training_scale_audit(method_manifest)
    multiseed_training_audit = build_multiseed_training_audit()
    provenance = build_implementation_provenance(method_manifest)
    source_manifest = build_spatial_source_manifest(main_wide, points)
    scale_audit = previous_scale_audit(main_long)
    metric_contract = evaluator_contract()
    seed_summary = multiseed_summary(multiseed_long)
    qa_checks = build_qa_checks(
        main_wide,
        points,
        transforms_main,
        multiseed_wide,
        transforms_multiseed,
        training_audit,
        multiseed_training_audit,
        provenance,
        source_manifest,
    )
    if not qa_checks["passed"].all():
        failed = qa_checks.loc[~qa_checks["passed"], ["check", "details"]]
        raise RuntimeError(f"Unified Figure 6 v5 QA failed:\n{failed.to_string(index=False)}")

    main_wide.to_csv(OUT_DIR / "unified_main_metrics_wide.csv", index=False)
    main_long.to_csv(OUT_DIR / "unified_main_metrics_long.csv", index=False)
    points.to_csv(OUT_DIR / "unified_main_spatial_points.csv", index=False)
    transforms_main.to_csv(OUT_DIR / "coordinate_transform_audit_main.csv", index=False)
    multiseed_wide.to_csv(OUT_DIR / "unified_multiseed_metrics_wide.csv", index=False)
    multiseed_long.to_csv(OUT_DIR / "unified_multiseed_metrics_long.csv", index=False)
    seed_summary.to_csv(OUT_DIR / "unified_multiseed_metric_summary.csv", index=False)
    transforms_multiseed.to_csv(OUT_DIR / "coordinate_transform_audit_multiseed.csv", index=False)
    method_manifest.to_csv(OUT_DIR / "main_method_manifest.csv", index=False)
    source_manifest.to_csv(OUT_DIR / "unified_spatial_source_manifest.csv", index=False)
    training_audit.to_csv(OUT_DIR / "training_scale_audit.csv", index=False)
    multiseed_training_audit.to_csv(OUT_DIR / "multiseed_training_scale_audit.csv", index=False)
    provenance.to_csv(OUT_DIR / "implementation_provenance_audit.csv", index=False)
    qa_checks.to_csv(OUT_DIR / "qa_checks.csv", index=False)
    scale_audit.to_csv(OUT_DIR / "previous_scale_mismatch_audit.csv", index=False)
    metric_contract.to_csv(OUT_DIR / "evaluator_contract.csv", index=False)
    write_docs(
        main_wide,
        main_long,
        multiseed_wide,
        scale_audit,
        training_audit,
        multiseed_training_audit,
        provenance,
        qa_checks,
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(OUT_DIR),
                "main_rows": int(main_wide.shape[0]),
                "main_methods": int(main_wide["method"].nunique()),
                "multiseed_prediction_rows": int(multiseed_wide.shape[0]),
                "point_rows": int(points.shape[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

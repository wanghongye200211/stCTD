#!/usr/bin/env python
"""Build an extended spatial-metric table for Figure 6c.

This is a read-only evaluator: it reuses the canonical Figure 6c prediction
files and observed target slices.  It does not train models or generate new
predictions.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import ConvexHull, cKDTree, distance


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_official_external_benchmark_v3 as prep  # noqa: E402
import run_official_external_baselines_v3 as runner  # noqa: E402


FIG6C_DIR = ROOT / "result_figures" / "figure6" / "c"
SOURCE_CSV = FIG6C_DIR / "figure6c_target_slice_metric_source.csv"
OUT_DIR = FIG6C_DIR / "extended_spatial_metrics_v1"
V4_DIR = ROOT / "result_figures" / "figure6" / "model_comparison_official_external_stvcr_ablation_v4"
V4_POINTS_CSV = V4_DIR / "v4_spatial_slice_prediction_points.csv"
MOUSE_LEGACY_POINTS_CSV = (
    ROOT
    / "result_figures"
    / "小鼠中脑"
    / "figure6_model_comparison_nohard_e400"
    / "figure6_nohard_e400_model_comparison_prediction_source.csv"
)

DATASET_TIME_LABELS = {
    "mouse_dorsal_midbrain": {1: "E14.5", 2: "E16.5"},
    "zesta_10_12_18": {1: "12 hpf", 2: "18 hpf"},
}

METHOD_ORDER = [
    "POT official solver",
    "moscot official",
    "Spateo official",
    "TrajectoryNet official",
    "PRESCIENT official",
    "Original stVCR",
    "SpaPOT",
    "SpaPOT without GAT",
    "SpaPOT + HJ",
]

MOUSE_LEGACY_METHOD_MAP = {
    "Original stVCR": "Original AE-hybrid",
    "SpaPOT without GAT": "AE10 hybrid",
}

MOUSE_TARGET_TIME_BY_LABEL = {
    1: 14.5,
    2: 16.5,
}

METRIC_DEFINITIONS = [
    ("spatial_chamfer", "Spatial Chamfer", "lower_better", "真实点云和预测点云之间的双向最近邻平均距离，越小表示局部空间位置越接近。"),
    ("spatial_iou_48", "Spatial IoU 48", "higher_better", "在统一 48-bin 网格上的空间占据交并比，越高表示整体形状重叠越好。"),
    ("occupancy_precision_48", "Occupancy precision 48", "higher_better", "预测占据网格中有多少落在真实占据区域内，反映预测是否外扩到错误区域。"),
    ("occupancy_recall_48", "Occupancy recall 48", "higher_better", "真实占据网格中有多少被预测覆盖，反映是否漏掉真实空间区域。"),
    ("occupancy_f1_48", "Occupancy F1 48", "higher_better", "占据 precision 和 recall 的调和平均，兼顾外扩和漏检。"),
    ("occupancy_jsd_48", "Occupancy JSD 48", "lower_better", "统一 48-bin 空间密度分布之间的 Jensen-Shannon divergence，越小表示密度分布越接近。"),
    ("boundary_f1_48_tol1", "Boundary F1 48 tol1", "higher_better", "网格边界 F1，允许 1 个网格单元误差，用来评估切片轮廓边界是否一致。"),
    ("centroid_distance", "Centroid distance", "lower_better", "预测和真实点云中心之间的欧氏距离，主要反映整体平移偏差。"),
    ("covariance_frobenius_error", "Covariance shape error", "lower_better", "二维空间协方差矩阵的归一化 Frobenius 误差，反映整体形状和尺度差异。"),
    ("principal_axis_angle_deg", "Principal-axis angle", "lower_better", "预测和真实主轴方向的角度差，单位为度，反映整体方向是否旋转偏移。"),
    ("anisotropy_error", "Anisotropy error", "lower_better", "空间协方差各向异性比例差异，反映是否过度拉长或压缩。"),
    ("convex_hull_area_ratio", "Convex-hull area ratio", "closer_to_1", "预测凸包面积除以真实凸包面积，接近 1 表示整体覆盖面积匹配。"),
    ("convex_hull_area_log2_error", "Convex-hull area log2 error", "lower_better", "预测和真实凸包面积比例的绝对 log2 误差，越小表示面积尺度越一致。"),
    ("hausdorff_distance", "Hausdorff distance", "lower_better", "双向最大最近邻距离，对极端离群点敏感。"),
    ("hausdorff95_distance", "Hausdorff95 distance", "lower_better", "双向 95 分位最近邻距离，比 Hausdorff 更稳健地反映边缘错位。"),
    ("pred_to_real_nn_median", "Pred-to-real NN median", "lower_better", "预测点到最近真实点距离的中位数，反映预测点局部落点精度。"),
    ("real_to_pred_nn_median", "Real-to-pred NN median", "lower_better", "真实点到最近预测点距离的中位数，反映真实区域是否被预测覆盖。"),
    ("pred_to_real_nn_p95", "Pred-to-real NN p95", "lower_better", "预测点到真实点最近邻距离的 95 分位，反映预测外溢程度。"),
    ("real_to_pred_nn_p95", "Real-to-pred NN p95", "lower_better", "真实点到预测点最近邻距离的 95 分位，反映真实区域漏检程度。"),
    ("coverage_precision_adaptive", "Adaptive coverage precision", "higher_better", "按真实切片局部间距自适应半径计算的预测覆盖精度，越高表示预测点更少落到错误区域。"),
    ("coverage_recall_adaptive", "Adaptive coverage recall", "higher_better", "按真实切片局部间距自适应半径计算的真实区域召回，越高表示真实区域被预测覆盖得更完整。"),
    ("sliced_wasserstein_128", "Sliced Wasserstein", "lower_better", "128 个固定方向投影后的平均一维 Wasserstein 距离，反映整体空间分布差异。"),
    ("energy_distance_sampled", "Energy distance", "lower_better", "采样多变量 energy distance，反映两个空间点分布的整体差异。"),
    ("mmd_rbf_sampled", "RBF MMD", "lower_better", "使用中位距离带宽的 RBF kernel MMD，反映分布级别差异。"),
    ("knn10_distance_wasserstein", "kNN10 density distance", "lower_better", "真实和预测点云 10 近邻距离分布的 Wasserstein 距离，反映局部密度结构差异。"),
]

COMPACT_METRICS = [
    "spatial_chamfer",
    "spatial_iou_48",
    "occupancy_f1_48",
    "boundary_f1_48_tol1",
    "centroid_distance",
    "hausdorff95_distance",
    "convex_hull_area_log2_error",
    "sliced_wasserstein_128",
    "knn10_distance_wasserstein",
]

POINT_DERIVED_METRICS = [
    metric
    for metric, _, _, _ in METRIC_DEFINITIONS
    if metric not in {"spatial_chamfer", "spatial_iou_48"}
]


def configure_cache_env() -> None:
    for key, value in prep.CACHE_ENV.items():
        os.environ.setdefault(key, value)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (999, method)


def target_time_display(dataset_id: str, target_label: int, target_time_input: float) -> str:
    return DATASET_TIME_LABELS.get(dataset_id, {}).get(int(target_label), f"{target_time_input:g}")


def load_point_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if V4_POINTS_CSV.exists():
        tables["v4"] = pd.read_csv(V4_POINTS_CSV)
    if MOUSE_LEGACY_POINTS_CSV.exists():
        tables["mouse_legacy"] = pd.read_csv(MOUSE_LEGACY_POINTS_CSV)
    return tables


def clean_xy(xy: np.ndarray) -> tuple[np.ndarray, int]:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64), 0
    xy = xy[:, :2]
    finite_mask = np.isfinite(xy).all(axis=1)
    return xy[finite_mask], int((~finite_mask).sum())


def prediction_xy(pred: ad.AnnData) -> tuple[np.ndarray | None, str]:
    if "X_state" in pred.obsm:
        state = np.asarray(pred.obsm["X_state"], dtype=np.float64)
        if state.ndim == 2 and state.shape[1] >= 2:
            xy, dropped = clean_xy(state[:, :2])
            source = "obsm:X_state[:,0:2]"
            if dropped:
                source = f"{source}; finite rows only, dropped={dropped}"
            return xy, source
    if "X_spatial_aligned" in pred.obsm:
        xy = np.asarray(pred.obsm["X_spatial_aligned"], dtype=np.float64)
        if xy.ndim == 2 and xy.shape[1] >= 2:
            xy, dropped = clean_xy(xy[:, :2])
            source = "obsm:X_spatial_aligned[:,0:2]"
            if dropped:
                source = f"{source}; finite rows only, dropped={dropped}"
            return xy, source
    return None, "missing spatial obsm"


def xy_from_columns(table: pd.DataFrame, x_col: str = "x", y_col: str = "y") -> np.ndarray | None:
    if table.empty or x_col not in table.columns or y_col not in table.columns:
        return None
    xy, _ = clean_xy(table[[x_col, y_col]].to_numpy(dtype=np.float64))
    if xy.shape[0] == 0:
        return None
    return xy


def point_source_xy(
    row: pd.Series,
    point_tables: dict[str, pd.DataFrame],
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    dataset_id = str(row["dataset_id"])
    method = str(row["method"])
    target_label = int(row["target_time_label"])

    mouse_legacy = point_tables.get("mouse_legacy")
    legacy_method = MOUSE_LEGACY_METHOD_MAP.get(method)
    if dataset_id == "mouse_dorsal_midbrain" and mouse_legacy is not None and legacy_method is not None:
        target_time = MOUSE_TARGET_TIME_BY_LABEL.get(target_label)
        legacy_target = mouse_legacy[(mouse_legacy["method"] == legacy_method) & np.isclose(mouse_legacy["time"].astype(float), float(target_time))]
        real = xy_from_columns(legacy_target[legacy_target["point_type"] == "observed_reference"])
        pred = xy_from_columns(legacy_target[legacy_target["point_type"] == "model_prediction"])
        if real is not None and pred is not None:
            return real, pred, "mouse_legacy_affine_prediction_source:x/y"

    v4 = point_tables.get("v4")
    if v4 is not None:
        v4_target = v4[(v4["dataset_id"] == dataset_id) & (v4["target_time_label"].astype(int) == target_label)]
        real = xy_from_columns(v4_target[v4_target["point_role"] == "observed_target"])
        pred = xy_from_columns(v4_target[(v4_target["method"] == method) & (v4_target["point_role"] == "predicted_reconstruction")])
        if real is not None and pred is not None:
            return real, pred, "v4_spatial_slice_prediction_points:x/y"

    return None, None, ""


def safe_convex_hull_area(xy: np.ndarray) -> float:
    if xy.shape[0] < 3:
        return float("nan")
    try:
        return float(ConvexHull(xy).volume)
    except Exception:
        return float("nan")


def covariance_features(real_xy: np.ndarray, pred_xy: np.ndarray) -> dict[str, float]:
    real_cov = np.cov(real_xy.T)
    pred_cov = np.cov(pred_xy.T)
    denom = max(float(np.linalg.norm(real_cov, ord="fro")), 1e-12)
    cov_err = float(np.linalg.norm(pred_cov - real_cov, ord="fro") / denom)

    def eig_info(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = np.maximum(vals[order], 0.0)
        vecs = vecs[:, order]
        return vals, vecs

    real_vals, real_vecs = eig_info(real_cov)
    pred_vals, pred_vecs = eig_info(pred_cov)
    real_axis = real_vecs[:, 0]
    pred_axis = pred_vecs[:, 0]
    cos_angle = float(np.clip(abs(np.dot(real_axis, pred_axis)), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_angle)))
    real_aniso = float(real_vals[0] / max(real_vals[1], 1e-12))
    pred_aniso = float(pred_vals[0] / max(pred_vals[1], 1e-12))
    return {
        "covariance_frobenius_error": cov_err,
        "principal_axis_angle_deg": angle,
        "real_anisotropy_ratio": real_aniso,
        "pred_anisotropy_ratio": pred_aniso,
        "anisotropy_error": abs(pred_aniso - real_aniso),
    }


def occupancy_arrays(real_xy: np.ndarray, pred_xy: np.ndarray, bins: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    combined = np.vstack([real_xy, pred_xy])
    lo = combined.min(axis=0)
    hi = combined.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    real_idx = np.floor((real_xy - lo) / span * (bins - 1)).astype(int)
    pred_idx = np.floor((pred_xy - lo) / span * (bins - 1)).astype(int)
    real_occ = np.zeros((bins, bins), dtype=bool)
    pred_occ = np.zeros((bins, bins), dtype=bool)
    real_occ[real_idx[:, 0], real_idx[:, 1]] = True
    pred_occ[pred_idx[:, 0], pred_idx[:, 1]] = True
    real_hist = np.zeros((bins, bins), dtype=float)
    pred_hist = np.zeros((bins, bins), dtype=float)
    np.add.at(real_hist, (real_idx[:, 0], real_idx[:, 1]), 1.0)
    np.add.at(pred_hist, (pred_idx[:, 0], pred_idx[:, 1]), 1.0)
    return real_occ, pred_occ, real_hist, pred_hist


def occupancy_metrics(real_xy: np.ndarray, pred_xy: np.ndarray) -> dict[str, float]:
    real_occ, pred_occ, real_hist, pred_hist = occupancy_arrays(real_xy, pred_xy, bins=48)
    inter = float(np.logical_and(real_occ, pred_occ).sum())
    pred_n = float(pred_occ.sum())
    real_n = float(real_occ.sum())
    union = float(np.logical_or(real_occ, pred_occ).sum())
    precision = inter / pred_n if pred_n else float("nan")
    recall = inter / real_n if real_n else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    iou = inter / union if union else float("nan")

    real_p = real_hist.ravel()
    pred_p = pred_hist.ravel()
    real_p = real_p / real_p.sum() if real_p.sum() else real_p
    pred_p = pred_p / pred_p.sum() if pred_p.sum() else pred_p
    mixture = 0.5 * (real_p + pred_p)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-300))))

    jsd = 0.5 * kl(real_p, mixture) + 0.5 * kl(pred_p, mixture)

    real_boundary = np.logical_and(real_occ, ~binary_erosion(real_occ))
    pred_boundary = np.logical_and(pred_occ, ~binary_erosion(pred_occ))
    real_boundary_dil = binary_dilation(real_boundary, iterations=1)
    pred_boundary_dil = binary_dilation(pred_boundary, iterations=1)
    bp = float(np.logical_and(pred_boundary, real_boundary_dil).sum()) / max(float(pred_boundary.sum()), 1.0)
    br = float(np.logical_and(real_boundary, pred_boundary_dil).sum()) / max(float(real_boundary.sum()), 1.0)
    bf1 = 2 * bp * br / (bp + br) if bp + br > 0 else float("nan")
    return {
        "spatial_iou_48": iou,
        "occupancy_precision_48": precision,
        "occupancy_recall_48": recall,
        "occupancy_f1_48": f1,
        "occupancy_jsd_48": jsd,
        "boundary_precision_48_tol1": bp,
        "boundary_recall_48_tol1": br,
        "boundary_f1_48_tol1": bf1,
        "real_occupied_cells_48": real_n,
        "pred_occupied_cells_48": pred_n,
    }


def nearest_neighbor_metrics(real_xy: np.ndarray, pred_xy: np.ndarray) -> dict[str, float]:
    real_tree = cKDTree(real_xy)
    pred_tree = cKDTree(pred_xy)
    d_pr, _ = real_tree.query(pred_xy, k=1)
    d_rp, _ = pred_tree.query(real_xy, k=1)
    chamfer = 0.5 * (float(np.mean(d_pr)) + float(np.mean(d_rp)))
    hausdorff = max(float(np.max(d_pr)), float(np.max(d_rp)))
    hausdorff95 = max(float(np.percentile(d_pr, 95)), float(np.percentile(d_rp, 95)))
    real_nn, _ = real_tree.query(real_xy, k=min(2, real_xy.shape[0]))
    if np.ndim(real_nn) == 2 and real_nn.shape[1] > 1:
        adaptive_radius = 2.0 * float(np.median(real_nn[:, 1]))
    else:
        adaptive_radius = float(np.median(d_rp))
    adaptive_radius = max(adaptive_radius, 1e-12)
    return {
        "spatial_chamfer": chamfer,
        "hausdorff_distance": hausdorff,
        "hausdorff95_distance": hausdorff95,
        "pred_to_real_nn_mean": float(np.mean(d_pr)),
        "real_to_pred_nn_mean": float(np.mean(d_rp)),
        "pred_to_real_nn_median": float(np.median(d_pr)),
        "real_to_pred_nn_median": float(np.median(d_rp)),
        "pred_to_real_nn_p95": float(np.percentile(d_pr, 95)),
        "real_to_pred_nn_p95": float(np.percentile(d_rp, 95)),
        "coverage_radius_adaptive": adaptive_radius,
        "coverage_precision_adaptive": float(np.mean(d_pr <= adaptive_radius)),
        "coverage_recall_adaptive": float(np.mean(d_rp <= adaptive_radius)),
    }


def quantile_resample(values: np.ndarray, n: int = 256) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return np.full(n, np.nan)
    if values.size == 1:
        return np.full(n, values[0])
    q_old = np.linspace(0.0, 1.0, values.size)
    q_new = np.linspace(0.0, 1.0, n)
    return np.interp(q_new, q_old, values)


def sliced_wasserstein(real_xy: np.ndarray, pred_xy: np.ndarray, n_dirs: int = 128, n_quantiles: int = 256) -> float:
    rng = np.random.default_rng(260704)
    angles = rng.uniform(0.0, np.pi, size=n_dirs)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    vals = []
    for direction in dirs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                real_proj = real_xy @ direction
                pred_proj = pred_xy @ direction
        real_q = quantile_resample(real_proj, n_quantiles)
        pred_q = quantile_resample(pred_proj, n_quantiles)
        diff = np.abs(real_q - pred_q)
        if np.isfinite(diff).any():
            vals.append(float(np.nanmean(diff)))
    return float(np.mean(vals)) if vals else float("nan")


def deterministic_sample(xy: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if xy.shape[0] <= max_n:
        return xy
    rng = np.random.default_rng(seed)
    idx = rng.choice(xy.shape[0], size=max_n, replace=False)
    return xy[np.sort(idx)]


def distribution_sampled_metrics(real_xy: np.ndarray, pred_xy: np.ndarray) -> dict[str, float]:
    real_s = deterministic_sample(real_xy, 700, 260704)
    pred_s = deterministic_sample(pred_xy, 700, 260705)
    rr = distance.cdist(real_s, real_s)
    pp = distance.cdist(pred_s, pred_s)
    rp = distance.cdist(real_s, pred_s)
    energy = float(2.0 * rp.mean() - rr.mean() - pp.mean())
    combined_dist = distance.pdist(np.vstack([real_s, pred_s]))
    bandwidth = float(np.median(combined_dist[combined_dist > 0])) if np.any(combined_dist > 0) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth * bandwidth, 1e-12)
    k_rr = np.exp(-gamma * rr * rr)
    k_pp = np.exp(-gamma * pp * pp)
    k_rp = np.exp(-gamma * rp * rp)
    mmd = float(k_rr.mean() + k_pp.mean() - 2.0 * k_rp.mean())

    def kth_dist(xy: np.ndarray, k: int = 10) -> np.ndarray:
        tree = cKDTree(xy)
        d, _ = tree.query(xy, k=min(k + 1, xy.shape[0]))
        if d.ndim == 1:
            return d
        return d[:, -1]

    real_knn = quantile_resample(kth_dist(real_xy, 10), 256)
    pred_knn = quantile_resample(kth_dist(pred_xy, 10), 256)
    knn_w = float(np.mean(np.abs(real_knn - pred_knn)))
    return {
        "sliced_wasserstein_128": sliced_wasserstein(real_xy, pred_xy),
        "energy_distance_sampled": energy,
        "mmd_rbf_sampled": mmd,
        "mmd_rbf_bandwidth": bandwidth,
        "knn10_distance_wasserstein": knn_w,
    }


def canonical_audit_metrics(real_xy: np.ndarray, pred_xy: np.ndarray) -> dict[str, float]:
    union_xy = np.vstack([real_xy, pred_xy])
    diag = float(np.linalg.norm(union_xy.max(axis=0) - union_xy.min(axis=0)))
    real_tree = cKDTree(real_xy)
    pred_tree = cKDTree(pred_xy)
    chamfer_sum = float(pred_tree.query(real_xy, k=1)[0].mean() + real_tree.query(pred_xy, k=1)[0].mean())

    grid_bins = 120
    x_edges = np.linspace(union_xy[:, 0].min(), union_xy[:, 0].max(), grid_bins + 1)
    y_edges = np.linspace(union_xy[:, 1].min(), union_xy[:, 1].max(), grid_bins + 1)
    real_grid = np.histogram2d(real_xy[:, 0], real_xy[:, 1], bins=[x_edges, y_edges])[0] > 0
    pred_grid = np.histogram2d(pred_xy[:, 0], pred_xy[:, 1], bins=[x_edges, y_edges])[0] > 0
    intersection = float(np.logical_and(real_grid, pred_grid).sum())
    union_count = float(np.logical_or(real_grid, pred_grid).sum())
    return {
        "canonical_audit_spatial_chamfer": float(chamfer_sum / diag if diag > 0 else np.nan),
        "canonical_audit_spatial_iou": float(intersection / union_count if union_count else np.nan),
    }


def spatial_extended_metrics(real_xy: np.ndarray, pred_xy: np.ndarray) -> dict[str, float]:
    real_xy, real_dropped = clean_xy(real_xy)
    pred_xy, pred_dropped = clean_xy(pred_xy)
    if real_xy.shape[0] == 0 or pred_xy.shape[0] == 0:
        return {}
    row: dict[str, float] = {}
    row["real_nonfinite_rows_dropped"] = float(real_dropped)
    row["pred_nonfinite_rows_dropped"] = float(pred_dropped)
    row.update(canonical_audit_metrics(real_xy, pred_xy))
    row.update(nearest_neighbor_metrics(real_xy, pred_xy))
    row.update(occupancy_metrics(real_xy, pred_xy))
    row["centroid_distance"] = float(np.linalg.norm(pred_xy.mean(axis=0) - real_xy.mean(axis=0)))
    row["real_centroid_x"] = float(real_xy[:, 0].mean())
    row["real_centroid_y"] = float(real_xy[:, 1].mean())
    row["pred_centroid_x"] = float(pred_xy[:, 0].mean())
    row["pred_centroid_y"] = float(pred_xy[:, 1].mean())
    row.update(covariance_features(real_xy, pred_xy))
    real_area = safe_convex_hull_area(real_xy)
    pred_area = safe_convex_hull_area(pred_xy)
    row["real_convex_hull_area"] = real_area
    row["pred_convex_hull_area"] = pred_area
    row["convex_hull_area_ratio"] = pred_area / real_area if np.isfinite(real_area) and real_area > 0 else float("nan")
    row["convex_hull_area_log2_error"] = (
        abs(math.log2(max(pred_area, 1e-300) / real_area)) if np.isfinite(real_area) and real_area > 0 and np.isfinite(pred_area) else float("nan")
    )
    row.update(distribution_sampled_metrics(real_xy, pred_xy))
    return row


def build_candidates(source: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "dataset_id",
        "dataset",
        "method",
        "target_time_input",
        "target_time_label",
        "prediction_path",
        "target_aware",
        "fully_generative",
        "alignment_only",
        "benchmark_role",
        "original_method",
        "metric_evaluator",
    ]
    candidates = source[keep_cols].drop_duplicates().copy()
    canonical = (
        source[source["metric"].isin(["annotation_l1", "spatial_chamfer", "spatial_iou"])]
        .pivot_table(
            index=["dataset_id", "method", "target_time_label"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "annotation_l1": "canonical_annotation_l1",
                "spatial_chamfer": "canonical_spatial_chamfer",
                "spatial_iou": "canonical_spatial_iou",
            }
        )
    )
    candidates = candidates.merge(canonical, on=["dataset_id", "method", "target_time_label"], how="left")
    candidates["method_order"] = candidates["method"].map(lambda item: method_sort_key(str(item))[0])
    return candidates.sort_values(["dataset_id", "method_order", "method", "target_time_label"]).drop(columns=["method_order"])


def finite_float(value: object) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f if np.isfinite(value_f) else float("nan")


def reconcile_with_canonical_spatial(base: dict[str, object], tolerance: float = 1e-8) -> dict[str, object]:
    canonical_chamfer = finite_float(base.get("canonical_spatial_chamfer"))
    canonical_iou = finite_float(base.get("canonical_spatial_iou"))
    source = str(base.get("spatial_source", ""))
    if source.startswith("mouse_legacy_affine_prediction_source"):
        audit_chamfer = finite_float(base.get("canonical_audit_spatial_chamfer"))
        audit_iou = finite_float(base.get("canonical_audit_spatial_iou"))
        point_chamfer = audit_chamfer if np.isfinite(audit_chamfer) else finite_float(base.get("spatial_chamfer"))
        point_iou = audit_iou if np.isfinite(audit_iou) else finite_float(base.get("spatial_iou_48"))
    else:
        point_chamfer = finite_float(base.get("spatial_chamfer"))
        point_iou = finite_float(base.get("spatial_iou_48"))

    base["point_recomputed_spatial_chamfer"] = point_chamfer
    base["point_recomputed_spatial_iou_48"] = point_iou

    chamfer_diff = abs(point_chamfer - canonical_chamfer) if np.isfinite(point_chamfer) and np.isfinite(canonical_chamfer) else float("nan")
    iou_diff = abs(point_iou - canonical_iou) if np.isfinite(point_iou) and np.isfinite(canonical_iou) else float("nan")
    base["canonical_spatial_chamfer_abs_diff"] = chamfer_diff
    base["canonical_spatial_iou_abs_diff"] = iou_diff

    has_canonical_spatial = np.isfinite(canonical_chamfer) or np.isfinite(canonical_iou)
    if has_canonical_spatial:
        if np.isfinite(canonical_chamfer):
            base["spatial_chamfer"] = canonical_chamfer
        if np.isfinite(canonical_iou):
            base["spatial_iou_48"] = canonical_iou
        if base.get("spatial_metric_status") not in {"computed", "spatial_unavailable"}:
            base["spatial_metric_status"] = "computed"

    chamfer_tolerance = tolerance
    iou_tolerance = tolerance
    if source.startswith("mouse_legacy_affine_prediction_source"):
        # The legacy Figure 6 source stores affine display coordinates.  They
        # audit very close to the canonical scalar metrics, but not bitwise
        # equal because the original evaluator used normalized aligned states.
        chamfer_tolerance = 0.005
        iou_tolerance = 0.03
    mismatch = (
        (np.isfinite(chamfer_diff) and chamfer_diff > chamfer_tolerance)
        or (np.isfinite(iou_diff) and iou_diff > iou_tolerance)
    )
    if mismatch:
        base["extended_spatial_metric_status"] = "coordinate_not_canonical"
        for metric in POINT_DERIVED_METRICS:
            base[metric] = float("nan")
    elif base.get("spatial_metric_status") == "computed":
        base["extended_spatial_metric_status"] = "computed"
    else:
        base["extended_spatial_metric_status"] = str(base.get("spatial_metric_status", "not_computed"))
    return base


def evaluate_candidate(row: pd.Series, bundle_cache: dict[str, runner.Bundle], point_tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    dataset_id = str(row["dataset_id"])
    if dataset_id not in bundle_cache:
        spec = next(item for item in prep.DATASETS if item.dataset_id == dataset_id)
        bundle_cache[dataset_id] = runner.load_bundle(spec)
    bundle = bundle_cache[dataset_id]
    target_label = int(row["target_time_label"])
    real_mask = bundle.labels == target_label
    real_xy = np.asarray(bundle.state[real_mask, :2], dtype=np.float64)
    pred_path = Path(str(row["prediction_path"]))
    base: dict[str, object] = row.to_dict()
    base["target_time_display"] = target_time_display(dataset_id, target_label, float(row["target_time_input"]))
    base["n_real"] = int(real_xy.shape[0])
    base["prediction_exists"] = pred_path.exists()
    point_real_xy, point_pred_xy, point_source = point_source_xy(row, point_tables)
    if point_real_xy is not None and point_pred_xy is not None:
        base["n_real"] = int(point_real_xy.shape[0])
        base["n_pred"] = int(point_pred_xy.shape[0])
        base["spatial_source"] = point_source
        base["spatial_metric_status"] = "computed"
        base.update(spatial_extended_metrics(point_real_xy, point_pred_xy))
        return reconcile_with_canonical_spatial(base)
    if not pred_path.exists():
        base.update({"spatial_metric_status": "missing_prediction_file", "n_pred": 0, "spatial_source": ""})
        return reconcile_with_canonical_spatial(base)
    pred = ad.read_h5ad(pred_path)
    pred_xy, spatial_source = prediction_xy(pred)
    base["spatial_source"] = spatial_source
    if pred_xy is None:
        base.update({"spatial_metric_status": "spatial_unavailable", "n_pred": int(pred.n_obs)})
        return reconcile_with_canonical_spatial(base)
    base["n_pred"] = int(pred_xy.shape[0])
    base["spatial_metric_status"] = "computed"
    base.update(spatial_extended_metrics(real_xy, pred_xy))
    return reconcile_with_canonical_spatial(base)


def write_metric_definitions() -> pd.DataFrame:
    definitions = pd.DataFrame(
        METRIC_DEFINITIONS,
        columns=["metric", "display_metric", "direction", "meaning_zh"],
    )
    definitions.to_csv(OUT_DIR / "figure6c_extended_spatial_metric_definitions.csv", index=False)
    lines = [
        "# Figure 6c extended spatial metric definitions",
        "",
        "| metric | direction | meaning |",
        "|---|---|---|",
    ]
    for item in definitions.itertuples(index=False):
        lines.append(f"| `{item.metric}` | `{item.direction}` | {item.meaning_zh} |")
    (OUT_DIR / "figure6c_extended_spatial_metric_definitions_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return definitions


def build_long_table(timepoint_wide: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "dataset_id",
        "dataset",
        "method",
        "target_time_display",
        "target_time_input",
        "target_time_label",
        "prediction_path",
        "spatial_metric_status",
        "extended_spatial_metric_status",
        "spatial_source",
        "n_real",
        "n_pred",
    ]
    metric_cols = [metric for metric in definitions["metric"] if metric in timepoint_wide.columns]
    long = timepoint_wide[id_cols + metric_cols].melt(
        id_vars=id_cols,
        var_name="metric",
        value_name="value",
    )
    long = long.merge(definitions, on="metric", how="left")
    return long


def build_method_mean(timepoint_wide: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [metric for metric in definitions["metric"] if metric in timepoint_wide.columns]
    group_cols = ["dataset_id", "dataset", "method", "spatial_metric_status", "extended_spatial_metric_status"]
    mean = timepoint_wide.groupby(group_cols, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    counts = (
        timepoint_wide.groupby(group_cols, dropna=False)
        .agg(
            target_timepoint_count=("target_time_label", "nunique"),
            computed_timepoint_count=("spatial_metric_status", lambda x: int((x == "computed").sum())),
            extended_computed_timepoint_count=("extended_spatial_metric_status", lambda x: int((x == "computed").sum())),
        )
        .reset_index()
    )
    return counts.merge(mean, on=group_cols, how="left")


def write_compact_tables(timepoint_wide: pd.DataFrame, method_mean: pd.DataFrame, definitions: pd.DataFrame) -> None:
    compact_metric_cols = [metric for metric in COMPACT_METRICS if metric in timepoint_wide.columns]
    meta_cols_timepoint = [
        "dataset_id",
        "dataset",
        "method",
        "target_time_display",
        "target_time_input",
        "target_time_label",
        "spatial_metric_status",
        "extended_spatial_metric_status",
        "target_aware",
        "fully_generative",
        "alignment_only",
        "benchmark_role",
        "prediction_path",
    ]
    meta_cols_mean = [
        "dataset_id",
        "dataset",
        "method",
        "spatial_metric_status",
        "extended_spatial_metric_status",
        "target_timepoint_count",
        "computed_timepoint_count",
        "extended_computed_timepoint_count",
    ]
    compact_timepoint = timepoint_wide[meta_cols_timepoint + compact_metric_cols].copy()
    compact_mean = method_mean[meta_cols_mean + compact_metric_cols].copy()
    compact_timepoint.to_csv(OUT_DIR / "figure6c_extended_spatial_metrics_compact_timepoint.csv", index=False)
    compact_mean.to_csv(OUT_DIR / "figure6c_extended_spatial_metrics_compact_method_mean.csv", index=False)

    direction = definitions.set_index("metric")["direction"].to_dict()
    display = definitions.set_index("metric")["display_metric"].to_dict()
    lines = [
        "# Figure 6c compact spatial metric table",
        "",
        "该表是从完整扩展指标表中挑出的论文友好版本；完整数值仍保存在 wide/long CSV。",
        "",
        "方向说明：`lower_better` 越低越好；`higher_better` 越高越好；`closer_to_1` 越接近 1 越好。",
        "",
    ]
    for dataset, part in compact_mean.groupby("dataset", sort=False):
        part = part.sort_values(["dataset_id", "method"], key=lambda col: col.map(lambda x: method_sort_key(str(x))[0]) if col.name == "method" else col)
        lines.extend([f"## {dataset}", ""])
        header = ["method", "canonical_status", "extended_status", "n_timepoints"] + [display.get(metric, metric) for metric in compact_metric_cols]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in part.itertuples(index=False):
            row_dict = row._asdict()
            values = [
                str(row_dict["method"]),
                str(row_dict["spatial_metric_status"]),
                str(row_dict["extended_spatial_metric_status"]),
                f"{int(row_dict['computed_timepoint_count'])}/{int(row_dict['target_timepoint_count'])}",
            ]
            for metric in compact_metric_cols:
                value = row_dict.get(metric)
                if pd.isna(value):
                    values.append("")
                elif direction.get(metric) == "higher_better":
                    values.append(f"{float(value):.3f}")
                else:
                    values.append(f"{float(value):.4g}")
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    (OUT_DIR / "figure6c_extended_spatial_metrics_compact_method_mean_zh.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(timepoint_wide: pd.DataFrame, method_mean: pd.DataFrame) -> None:
    status_counts = timepoint_wide["spatial_metric_status"].value_counts(dropna=False).to_dict()
    extended_status_counts = timepoint_wide["extended_spatial_metric_status"].value_counts(dropna=False).to_dict()
    computed_methods = method_mean[method_mean["spatial_metric_status"].eq("computed")].shape[0]
    text = f"""# Figure 6c extended spatial metric table

This package evaluates additional spatial-quality metrics from the existing
canonical Figure 6c prediction files. It does not train models or create new
predictions.

Input source:
`{SOURCE_CSV}`

Outputs:
- `figure6c_extended_spatial_metrics_timepoint_wide.csv`: one row per dataset/method/target time point.
- `figure6c_extended_spatial_metrics_timepoint_long.csv`: long metric table with metric meanings and directions.
- `figure6c_extended_spatial_metrics_method_mean_wide.csv`: mean over the two target time points for each dataset/method.
- `figure6c_extended_spatial_metrics_compact_timepoint.csv`: selected spatial metrics for easier review.
- `figure6c_extended_spatial_metrics_compact_method_mean.csv`: selected spatial metrics averaged over the two target time points.
- `figure6c_extended_spatial_metrics_compact_method_mean_zh.md`: readable compact table.
- `figure6c_extended_spatial_metric_definitions.csv`: metric direction and interpretation.
- `figure6c_extended_spatial_metric_definitions_zh.md`: readable metric notes.

Status counts:
```json
{json.dumps(status_counts, ensure_ascii=False, indent=2)}
```

Extended metric status counts:
```json
{json.dumps(extended_status_counts, ensure_ascii=False, indent=2)}
```

Computed dataset/method groups: {computed_methods}

Notes:
- `spatial_iou_48` uses the same 48-bin occupancy convention as the existing Figure 6 evaluator.
- `spatial_chamfer` and `spatial_iou_48` are forced to the canonical Figure 6c source values.
- Extra point-derived metrics are left blank when the available point coordinates do not reproduce canonical Chamfer/IoU.
- PRESCIENT official predictions are expression/PCA trajectories without spatial coordinates here, so spatial metrics are marked `spatial_unavailable`.
- POT, moscot and Spateo are target-aware baselines; the table keeps their flags from the Figure 6 source.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_cache_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_CSV)
    candidates = build_candidates(source)
    bundle_cache: dict[str, runner.Bundle] = {}
    point_tables = load_point_tables()
    rows = [evaluate_candidate(row, bundle_cache, point_tables) for _, row in candidates.iterrows()]
    timepoint_wide = pd.DataFrame(rows)
    definitions = write_metric_definitions()
    long = build_long_table(timepoint_wide, definitions)
    method_mean = build_method_mean(timepoint_wide, definitions)
    timepoint_wide.to_csv(OUT_DIR / "figure6c_extended_spatial_metrics_timepoint_wide.csv", index=False)
    long.to_csv(OUT_DIR / "figure6c_extended_spatial_metrics_timepoint_long.csv", index=False)
    method_mean.to_csv(OUT_DIR / "figure6c_extended_spatial_metrics_method_mean_wide.csv", index=False)
    write_compact_tables(timepoint_wide, method_mean, definitions)
    write_readme(timepoint_wide, method_mean)
    print(
        json.dumps(
            {
                "status": "built_extended_spatial_metric_table",
                "timepoint_rows": int(timepoint_wide.shape[0]),
                "method_mean_rows": int(method_mean.shape[0]),
                "out_dir": str(OUT_DIR),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

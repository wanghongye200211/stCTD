#!/usr/bin/env python
"""Run official target-aware external baselines for the v3 benchmark.

This script contains evaluation and file-writing glue, but the baseline
transport/alignment itself must come from official packages:

- POT: ``ot`` official package.
- moscot: ``moscot.problems.time.TemporalProblem``.
- Spateo: ``spateo.alignment``.

It deliberately does not include the previous local VelocityNet or potential
surrogate baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_official_external_benchmark_v3 as prep  # noqa: E402


OUT_DIR = prep.OUT_DIR


@dataclass
class Bundle:
    spec: prep.DatasetSpec
    adata: ad.AnnData
    state: np.ndarray
    state_z: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    labels: np.ndarray
    time_mapping: dict[float, int]
    inverse_time_mapping: dict[int, float]
    annotation_scaler: StandardScaler
    annotation_classifier: KNeighborsClassifier
    detailed_classifier: KNeighborsClassifier | None


def configure_cache_env() -> None:
    for key, value in prep.CACHE_ENV.items():
        os.environ.setdefault(key, value)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def selected_specs(dataset_id: str) -> list[prep.DatasetSpec]:
    if dataset_id == "all":
        return list(prep.DATASETS)
    for spec in prep.DATASETS:
        if spec.dataset_id == dataset_id:
            return [spec]
    raise ValueError(f"Unknown dataset_id: {dataset_id}")


def load_bundle(spec: prep.DatasetSpec) -> Bundle:
    import scanpy as sc

    adata = sc.read_h5ad(spec.observed_h5ad)
    state = prep.state_matrix(adata).astype(np.float32)
    state_z, mean, std = prep.standardized_state(state)
    labels, mapping = prep.consecutive_time_labels(adata.obs[spec.time_input_key].to_numpy())
    inverse = {idx: original for original, idx in mapping.items()}
    max_time = max(float(t) for t in adata.obs[spec.time_input_key].to_numpy())
    time_feature = (adata.obs[spec.time_input_key].to_numpy(dtype=float) / max(max_time, 1e-8))[:, None]
    classifier_features = np.hstack([state, time_feature]).astype(np.float32)
    scaler = StandardScaler().fit(classifier_features)
    classifier_features = scaler.transform(classifier_features)
    annotation_classifier = KNeighborsClassifier(n_neighbors=31, weights="distance")
    annotation_classifier.fit(classifier_features, adata.obs[spec.annotation_key].astype(str).to_numpy())
    detailed_classifier = None
    if spec.detailed_key and spec.detailed_key in adata.obs:
        detailed_classifier = KNeighborsClassifier(n_neighbors=31, weights="distance")
        detailed_classifier.fit(classifier_features, adata.obs[spec.detailed_key].astype(str).to_numpy())
    return Bundle(
        spec=spec,
        adata=adata,
        state=state,
        state_z=state_z,
        state_mean=mean,
        state_std=std,
        labels=labels,
        time_mapping=mapping,
        inverse_time_mapping=inverse,
        annotation_scaler=scaler,
        annotation_classifier=annotation_classifier,
        detailed_classifier=detailed_classifier,
    )


def label_predictions(bundle: Bundle, states: np.ndarray, time_input: float, detailed: bool = False) -> np.ndarray:
    max_time = max(float(t) for t in bundle.adata.obs[bundle.spec.time_input_key].to_numpy())
    time_feature = np.full((states.shape[0], 1), float(time_input) / max(max_time, 1e-8), dtype=np.float32)
    features = np.hstack([states.astype(np.float32), time_feature])
    features = bundle.annotation_scaler.transform(features)
    if detailed:
        if bundle.detailed_classifier is None:
            raise ValueError("No detailed annotation classifier for this dataset.")
        return bundle.detailed_classifier.predict(features)
    return bundle.annotation_classifier.predict(features)


def composition_metrics(real: np.ndarray, pred: np.ndarray, prefix: str = "annotation") -> dict[str, float]:
    cats = sorted(set(real.astype(str)).union(set(pred.astype(str))))
    real_counts = pd.Series(real.astype(str)).value_counts(normalize=True)
    pred_counts = pd.Series(pred.astype(str)).value_counts(normalize=True)
    real_vec = np.array([real_counts.get(cat, 0.0) for cat in cats], dtype=float)
    pred_vec = np.array([pred_counts.get(cat, 0.0) for cat in cats], dtype=float)
    l1 = 0.5 * float(np.abs(real_vec - pred_vec).sum())
    if len(cats) > 1 and np.std(real_vec) > 0 and np.std(pred_vec) > 0:
        corr = float(np.corrcoef(real_vec, pred_vec)[0, 1])
    else:
        corr = float("nan")
    return {f"{prefix}_l1": l1, f"{prefix}_corr": corr}


def spatial_metrics(real_xy: np.ndarray, pred_xy: np.ndarray) -> dict[str, float]:
    if len(real_xy) == 0 or len(pred_xy) == 0:
        return {"spatial_chamfer": float("nan"), "spatial_iou": float("nan")}
    real_tree = cKDTree(real_xy)
    pred_tree = cKDTree(pred_xy)
    d_pr, _ = real_tree.query(pred_xy, k=1)
    d_rp, _ = pred_tree.query(real_xy, k=1)
    chamfer = 0.5 * (float(np.mean(d_pr)) + float(np.mean(d_rp)))

    combined = np.vstack([real_xy, pred_xy])
    lo = combined.min(axis=0)
    hi = combined.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    bins = 48
    real_idx = np.floor((real_xy - lo) / span * (bins - 1)).astype(int)
    pred_idx = np.floor((pred_xy - lo) / span * (bins - 1)).astype(int)
    real_occ = np.zeros((bins, bins), dtype=bool)
    pred_occ = np.zeros((bins, bins), dtype=bool)
    real_occ[real_idx[:, 0], real_idx[:, 1]] = True
    pred_occ[pred_idx[:, 0], pred_idx[:, 1]] = True
    union = np.logical_or(real_occ, pred_occ).sum()
    iou = float(np.logical_and(real_occ, pred_occ).sum() / union) if union else float("nan")
    return {"spatial_chamfer": chamfer, "spatial_iou": iou}


def evaluate_states(bundle: Bundle, method: str, target_label: int, states: np.ndarray, status: str) -> dict[str, object]:
    target_time = bundle.inverse_time_mapping[target_label]
    mask = bundle.labels == target_label
    real_states = bundle.state[mask]
    real_anno = bundle.adata.obs[bundle.spec.annotation_key].astype(str).to_numpy()[mask]
    pred_anno = label_predictions(bundle, states, target_time, detailed=False)
    row: dict[str, object] = {
        "dataset_id": bundle.spec.dataset_id,
        "dataset": bundle.spec.display_name,
        "method": method,
        "target_time_input": target_time,
        "target_time_label": target_label,
        "n_real": int(real_states.shape[0]),
        "n_pred": int(states.shape[0]),
        "official_status": status,
    }
    row.update(composition_metrics(real_anno, pred_anno, "annotation"))
    if bundle.detailed_classifier is not None and bundle.spec.detailed_key:
        real_detail = bundle.adata.obs[bundle.spec.detailed_key].astype(str).to_numpy()[mask]
        pred_detail = label_predictions(bundle, states, target_time, detailed=True)
        row.update(composition_metrics(real_detail, pred_detail, "detailed"))
    row.update(spatial_metrics(real_states[:, :2], states[:, :2]))
    return row


def save_prediction(bundle: Bundle, method_dir: Path, method: str, target_label: int, states: np.ndarray) -> Path:
    target_time = bundle.inverse_time_mapping[target_label]
    pred_anno = label_predictions(bundle, states, target_time, detailed=False)
    obs = pd.DataFrame(
        {
            "time_input": np.full(states.shape[0], target_time),
            "time": np.full(states.shape[0], target_time),
            "Annotation": pred_anno.astype(str),
            "method": method,
            "target_aware": True,
        },
        index=[f"{bundle.spec.dataset_id}_{method}_{target_label}_{i}" for i in range(states.shape[0])],
    )
    if bundle.detailed_classifier is not None and bundle.spec.detailed_key:
        obs[bundle.spec.detailed_key] = label_predictions(bundle, states, target_time, detailed=True).astype(str)
    gene = states[:, 2:].astype(np.float32)
    pred = ad.AnnData(X=gene, obs=obs)
    pred.obsm["X_spatial_aligned"] = states[:, :2].astype(np.float32)
    pred.obsm["X_state"] = states.astype(np.float32)
    pred.uns["official_baseline"] = {
        "method": method,
        "dataset_id": bundle.spec.dataset_id,
        "target_time_input": target_time,
        "target_aware": True,
        "note": "Generated by official external-baseline runner; method API is official, evaluation glue is project code.",
    }
    pred_path = method_dir / "predictions" / f"predict_time_label_{target_label}.h5ad"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(pred_path)
    return pred_path


def run_pot(bundle: Bundle, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import ot

    method = "POT official solver"
    method_dir = OUT_DIR / "official_runs" / bundle.spec.dataset_id / "POT_official_solver"
    method_dir.mkdir(parents=True, exist_ok=True)
    source_label = int(bundle.labels.min())
    source_mask = bundle.labels == source_label
    source_z = bundle.state_z[source_mask]
    source_state = bundle.state[source_mask]
    metrics: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for target_label in sorted(np.unique(bundle.labels)):
        target_label = int(target_label)
        started = time.time()
        if target_label == source_label:
            pred_state = source_state.copy()
            status = "source_observed_passthrough"
            convergence_note = ""
        else:
            target_mask = bundle.labels == target_label
            target_z = bundle.state_z[target_mask]
            target_state = bundle.state[target_mask]
            cost = ot.dist(source_z, target_z, metric="sqeuclidean").astype(np.float64)
            positive = cost[cost > 0]
            if positive.size:
                cost = cost / float(np.median(positive))
            a = np.full(source_z.shape[0], 1.0 / source_z.shape[0], dtype=np.float64)
            b = np.full(target_z.shape[0], 1.0 / target_z.shape[0], dtype=np.float64)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if args.pot_solver == "sinkhorn_log":
                    transport = ot.bregman.sinkhorn_log(
                        a,
                        b,
                        cost,
                        reg=args.pot_reg,
                        numItermax=args.pot_max_iter,
                        stopThr=args.pot_stop_thr,
                        verbose=False,
                    )
                    official_api = "ot.bregman.sinkhorn_log"
                else:
                    transport = ot.sinkhorn(
                        a,
                        b,
                        cost,
                        reg=args.pot_reg,
                        numItermax=args.pot_max_iter,
                        stopThr=args.pot_stop_thr,
                        verbose=False,
                    )
                    official_api = "ot.sinkhorn"
            row_sum = np.maximum(transport.sum(axis=1, keepdims=True), 1e-12)
            pred_state = (transport @ target_state) / row_sum
            if not np.isfinite(pred_state).all():
                raise FloatingPointError(
                    f"{official_api} produced non-finite barycentric states for "
                    f"{bundle.spec.dataset_id} target label {target_label}."
                )
            status = "completed_official_pot_sinkhorn"
            convergence_note = "; ".join(str(item.message) for item in caught)
        pred_path = save_prediction(bundle, method_dir, method, target_label, pred_state.astype(np.float32))
        metric_row = evaluate_states(bundle, method, target_label, pred_state.astype(np.float32), status)
        metric_row["prediction_path"] = str(pred_path)
        metrics.append(metric_row)
        audit.append(
            {
                "dataset_id": bundle.spec.dataset_id,
                "method": method,
                "target_time_label": target_label,
                "official_package": "POT",
                "official_api": "source_observed_passthrough" if target_label == source_label else official_api,
                "n_source": int(source_state.shape[0]),
                "n_target": int(np.sum(bundle.labels == target_label)),
                "reg": args.pot_reg,
                "max_iter": args.pot_max_iter,
                "runtime_seconds": round(time.time() - started, 3),
                "status": status,
                "warnings": convergence_note,
            }
        )
    return metrics, audit


def run_moscot(bundle: Bundle, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from moscot.problems.time import TemporalProblem

    method = "moscot official"
    method_dir = OUT_DIR / "official_runs" / bundle.spec.dataset_id / "moscot_official"
    method_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(X=bundle.state_z.astype(np.float32), obs=pd.DataFrame({"time_label": bundle.labels.astype(int)}))
    adata.obsm["X_state"] = bundle.state_z.astype(np.float32)
    started = time.time()
    tp = TemporalProblem(adata).prepare(
        time_key="time_label",
        joint_attr="X_state",
        policy="triu",
        cost="sq_euclidean",
    )
    tp.solve(
        epsilon=args.moscot_epsilon,
        max_iterations=args.moscot_max_iterations,
        batch_size=args.moscot_batch_size,
        device="cpu",
    )
    source_label = int(bundle.labels.min())
    source_mask = bundle.labels == source_label
    source_state = bundle.state[source_mask]
    metrics: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for target_label in sorted(np.unique(bundle.labels)):
        target_label = int(target_label)
        target_started = time.time()
        if target_label == source_label:
            pred_state = source_state.copy()
            status = "source_observed_passthrough"
        else:
            target_state = bundle.state[bundle.labels == target_label]
            numerator = tp.pull(
                source=source_label,
                target=target_label,
                data=target_state,
                key_added=None,
                return_all=False,
                normalize=False,
                scale_by_marginals=False,
            )
            denominator = tp.pull(
                source=source_label,
                target=target_label,
                data=np.ones((target_state.shape[0], 1), dtype=np.float32),
                key_added=None,
                return_all=False,
                normalize=False,
                scale_by_marginals=False,
            )
            numerator = np.asarray(numerator, dtype=np.float64)
            denominator = np.asarray(denominator, dtype=np.float64)
            pred_state = numerator / np.maximum(denominator, 1e-12)
            if not np.isfinite(pred_state).all():
                raise FloatingPointError(
                    f"moscot barycentric readout produced non-finite states for "
                    f"{bundle.spec.dataset_id} target label {target_label}."
                )
            pred_state = pred_state.astype(np.float32)
            status = "completed_official_moscot_temporalproblem"
        pred_path = save_prediction(bundle, method_dir, method, target_label, pred_state)
        metric_row = evaluate_states(bundle, method, target_label, pred_state, status)
        metric_row["prediction_path"] = str(pred_path)
        metrics.append(metric_row)
        audit.append(
            {
                "dataset_id": bundle.spec.dataset_id,
                "method": method,
                "target_time_label": target_label,
                "official_package": "moscot",
                "official_api": "moscot.problems.time.TemporalProblem.prepare/solve/pull",
                "n_source": int(source_state.shape[0]),
                "n_target": int(np.sum(bundle.labels == target_label)),
                "epsilon": args.moscot_epsilon,
                "max_iterations": args.moscot_max_iterations,
                "batch_size": args.moscot_batch_size,
                "readout": "row_normalized_coupling_barycentric_target_state",
                "normalize_input_features": False,
                "scale_by_marginals": False,
                "runtime_seconds": round(time.time() - target_started, 3),
                "overall_solve_seconds": round(time.time() - started, 3),
                "status": status,
                "warnings": "",
            }
        )
    return metrics, audit


def make_spateo_model(bundle: Bundle, label: int):
    mask = bundle.labels == label
    model = ad.AnnData(
        X=prep.as_dense(bundle.adata.X[mask]).astype(np.float32),
        obs=bundle.adata.obs.iloc[np.flatnonzero(mask)].copy(),
        var=bundle.adata.var.copy(),
    )
    model.obsm["spatial"] = bundle.state[mask, :2].astype(np.float32)
    model.obsm["X_state"] = bundle.state[mask].astype(np.float32)
    model.uns["__type"] = "UMI"
    return model


def run_spateo(bundle: Bundle, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import spateo.alignment as alignment

    method = "Spateo official"
    method_dir = OUT_DIR / "official_runs" / bundle.spec.dataset_id / "Spateo_official"
    method_dir.mkdir(parents=True, exist_ok=True)
    source_label = int(bundle.labels.min())
    source_state = bundle.state[bundle.labels == source_label]
    metrics: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for target_label in sorted(np.unique(bundle.labels)):
        target_label = int(target_label)
        started = time.time()
        if target_label == source_label:
            pred_state = source_state.copy()
            status = "source_observed_passthrough"
            warning_text = ""
        else:
            source_model = make_spateo_model(bundle, source_label)
            target_model = make_spateo_model(bundle, target_label)
            genes = None
            if args.spateo_max_genes and source_model.n_vars > args.spateo_max_genes:
                genes = list(source_model.var_names[: args.spateo_max_genes])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if args.spateo_algorithm == "morpho":
                    aligned_models, _pis = alignment.morpho_align(
                        [target_model, source_model],
                        rep_layer="X",
                        rep_field="layer",
                        genes=genes,
                        spatial_key="spatial",
                        key_added="align_spatial",
                        iter_key_added=None,
                        vecfld_key_added=None,
                        mode="SN-S",
                        dissimilarity="kl",
                        max_iter=args.spateo_num_iter_max,
                        device="cpu",
                        verbose=True,
                    )
                    aligned_source = aligned_models[1]
                    official_api = "spateo.alignment.morpho_align"
                    status = "completed_official_spateo_morpho_align"
                else:
                    aligned_models, _pis = alignment.paste_align(
                        [source_model, target_model],
                        layer="X",
                        genes=genes,
                        spatial_key="spatial",
                        key_added="align_spatial",
                        alpha=args.spateo_alpha,
                        numItermax=args.spateo_num_iter_max,
                        numItermaxEmd=args.spateo_num_iter_max_emd,
                        device="cpu",
                        verbose=True,
                    )
                    aligned_source = aligned_models[0]
                    official_api = "spateo.alignment.paste_align"
                    status = "completed_official_spateo_paste_align"
            aligned_xy = np.asarray(aligned_source.obsm["align_spatial"], dtype=np.float32)
            pred_state = source_state.copy()
            pred_state[:, :2] = aligned_xy[:, :2]
            warning_text = "; ".join(str(item.message) for item in caught)
        pred_path = save_prediction(bundle, method_dir, method, target_label, pred_state.astype(np.float32))
        metric_row = evaluate_states(bundle, method, target_label, pred_state.astype(np.float32), status)
        metric_row["prediction_path"] = str(pred_path)
        metrics.append(metric_row)
        audit.append(
            {
                "dataset_id": bundle.spec.dataset_id,
                "method": method,
                "target_time_label": target_label,
                "official_package": "spateo-release",
                "official_api": "source_observed_passthrough" if target_label == source_label else official_api,
                "n_source": int(source_state.shape[0]),
                "n_target": int(np.sum(bundle.labels == target_label)),
                "algorithm": args.spateo_algorithm,
                "alpha": args.spateo_alpha,
                "numItermax": args.spateo_num_iter_max,
                "numItermaxEmd": args.spateo_num_iter_max_emd,
                "runtime_seconds": round(time.time() - started, 3),
                "status": status,
                "warnings": warning_text,
            }
        )
    return metrics, audit


def update_manifest_status(dataset_id: str, method: str, metrics: list[dict[str, object]]) -> None:
    manifest_path = OUT_DIR / "official_method_manifest.csv"
    if not manifest_path.exists() or not metrics:
        return
    manifest = pd.read_csv(manifest_path)
    pred_paths = [str(row.get("prediction_path", "")) for row in metrics if row.get("prediction_path")]
    mask = (manifest["dataset_id"] == dataset_id) & (manifest["method"] == method)
    manifest.loc[mask, "official_status"] = "completed_official_run"
    manifest.loc[mask, "qualified_for_main_figure_now"] = True
    manifest.loc[mask, "exclusion_reason"] = ""
    manifest.loc[mask, "prediction_path"] = ";".join(pred_paths)
    manifest.to_csv(manifest_path, index=False)


def append_outputs(dataset_id: str, method: str, metrics: list[dict[str, object]], audit: list[dict[str, object]]) -> None:
    method_dir = OUT_DIR / "official_runs" / dataset_id / prep.safe_name(method)
    method_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    audit_df = pd.DataFrame(audit)
    metrics_path = method_dir / "official_metrics_by_time.csv"
    audit_path = method_dir / "official_method_run_audit.csv"
    metrics_df.to_csv(metrics_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    report = {
        "dataset_id": dataset_id,
        "method": method,
        "metrics_by_time": str(metrics_path),
        "run_audit": str(audit_path),
        "n_rows": int(metrics_df.shape[0]),
    }
    (method_dir / "official_method_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    update_manifest_status(dataset_id, method, metrics)


def parse_methods(methods: str) -> list[str]:
    values = [item.strip().lower() for item in methods.replace(",", " ").split() if item.strip()]
    if "all" in values:
        return ["pot", "moscot", "spateo"]
    allowed = {"pot", "moscot", "spateo"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="all", help="Dataset id or 'all'.")
    parser.add_argument("--methods", default="pot", help="Comma/space list: pot, moscot, spateo, all.")
    parser.add_argument("--pot-reg", type=float, default=0.05)
    parser.add_argument("--pot-solver", choices=["sinkhorn_log", "sinkhorn"], default="sinkhorn_log")
    parser.add_argument("--pot-max-iter", type=int, default=5000)
    parser.add_argument("--pot-stop-thr", type=float, default=1e-9)
    parser.add_argument("--moscot-epsilon", type=float, default=0.05)
    parser.add_argument("--moscot-max-iterations", type=int, default=10000)
    parser.add_argument("--moscot-batch-size", type=int, default=None)
    parser.add_argument("--spateo-alpha", type=float, default=0.1)
    parser.add_argument("--spateo-algorithm", choices=["morpho", "paste"], default="morpho")
    parser.add_argument("--spateo-num-iter-max", type=int, default=200)
    parser.add_argument("--spateo-num-iter-max-emd", type=int, default=100000)
    parser.add_argument(
        "--spateo-max-genes",
        type=int,
        default=None,
        help="Optional memory-safety cap. Leave unset for all genes in formal runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cache_env()
    methods = parse_methods(args.methods)
    results: list[dict[str, object]] = []
    for spec in selected_specs(args.dataset_id):
        bundle = load_bundle(spec)
        for method in methods:
            if method == "pot":
                metrics, audit = run_pot(bundle, args)
                method_name = "POT official solver"
            elif method == "moscot":
                metrics, audit = run_moscot(bundle, args)
                method_name = "moscot official"
            elif method == "spateo":
                metrics, audit = run_spateo(bundle, args)
                method_name = "Spateo official"
            else:
                raise AssertionError(method)
            append_outputs(spec.dataset_id, method_name, metrics, audit)
            results.append({"dataset_id": spec.dataset_id, "method": method_name, "n_metrics": len(metrics)})
    print(json.dumps({"completed": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()

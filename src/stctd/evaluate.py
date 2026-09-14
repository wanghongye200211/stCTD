from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import DataConfig, TrainConfig
from .data import PreparedData
from .fields import STCTDModel
from .integrator import rollout_stctd
from .latent_classifier import LatentClassifierConfig, TrainedLatentClassifier, predict_latent_labels, train_latent_classifier
from .spatiotemporal_classifier import create_spatiotemporal_classifier
from embedding.preprocessing.ae_checkpoint import decode_gene_latent, load_frozen_decoder


@dataclass
class DecoderRuntime:
    bundle: Any
    latent_mean: torch.Tensor | None
    latent_std: torch.Tensor | None

    def to_decoder_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None or self.latent_std is None:
            return latent
        return latent * self.latent_std + self.latent_mean


def load_decoder_runtime(checkpoint_path: Path, device: torch.device) -> DecoderRuntime:
    bundle = load_frozen_decoder(checkpoint_path, device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    latent_mean = None
    latent_std = None
    if isinstance(payload, dict) and payload.get("model_type") == "gene_prior_gatae":
        latent_mean = torch.as_tensor(payload["latent_mean"], dtype=torch.float32, device=device).reshape(1, -1)
        latent_std = torch.as_tensor(payload["latent_std"], dtype=torch.float32, device=device).reshape(1, -1)
    return DecoderRuntime(bundle=bundle, latent_mean=latent_mean, latent_std=latent_std)


def _decode_pred_expression(
    pred_state: torch.Tensor,
    data: PreparedData,
    decoder: DecoderRuntime,
) -> torch.Tensor:
    scaled_latent = pred_state[:, data.spatial_dim :]
    mean = torch.as_tensor(data.scaler.latent_mean, dtype=scaled_latent.dtype, device=scaled_latent.device).reshape(1, -1)
    std = torch.as_tensor(data.scaler.latent_std, dtype=scaled_latent.dtype, device=scaled_latent.device).reshape(1, -1)
    latent = scaled_latent * std + mean
    decoder_latent = decoder.to_decoder_latent(latent)
    return decode_gene_latent(decoder.bundle, decoder_latent)


def _color_map(adata: ad.AnnData, annotation_key: str) -> dict[str, Any]:
    categories = list(adata.obs[annotation_key].cat.categories)
    colors = list(adata.uns.get(f"{annotation_key}_colors", adata.uns.get("Annotation_colors", [])))
    if len(colors) != len(categories):
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(categories))]
    return dict(zip(categories, colors))


def _label_metrics(real: ad.AnnData, pred: ad.AnnData, annotation_key: str) -> dict[str, Any]:
    categories = sorted(set(real.obs[annotation_key].astype(str)) | set(pred.obs[annotation_key].astype(str)))
    real_counts = real.obs[annotation_key].astype(str).value_counts().reindex(categories, fill_value=0)
    pred_counts = pred.obs[annotation_key].astype(str).value_counts().reindex(categories, fill_value=0)
    real_prop = (real_counts / real_counts.sum()).to_numpy(dtype=np.float32)
    pred_prop = (pred_counts / pred_counts.sum()).to_numpy(dtype=np.float32)
    corr = float(np.corrcoef(real_prop, pred_prop)[0, 1]) if len(categories) > 1 else 1.0
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "real_n": int(real.n_obs),
        "pred_n": int(pred.n_obs),
        "cell_delta": int(pred.n_obs - real.n_obs),
        "cell_ratio": float(pred.n_obs / real.n_obs),
        "label_prop_corr": corr,
        "label_prop_l1": float(np.abs(real_prop - pred_prop).sum()),
        "real_counts": {str(k): int(v) for k, v in real_counts.items()},
        "pred_counts": {str(k): int(v) for k, v in pred_counts.items()},
    }


def _spatial_metrics(real: ad.AnnData, pred: ad.AnnData, annotation_key: str, spatial_key: str) -> dict[str, Any]:
    real_xy = np.asarray(real.obsm[spatial_key], dtype=np.float32)
    pred_xy = np.asarray(pred.obsm[spatial_key], dtype=np.float32)
    union = np.vstack([real_xy, pred_xy])
    diag = float(np.linalg.norm(union.max(axis=0) - union.min(axis=0)))
    real_tree = cKDTree(real_xy)
    pred_tree = cKDTree(pred_xy)
    chamfer = float(pred_tree.query(real_xy, k=1)[0].mean() + real_tree.query(pred_xy, k=1)[0].mean())
    grid_bins = 120
    x_edges = np.linspace(union[:, 0].min(), union[:, 0].max(), grid_bins + 1)
    y_edges = np.linspace(union[:, 1].min(), union[:, 1].max(), grid_bins + 1)
    real_grid = np.histogram2d(real_xy[:, 0], real_xy[:, 1], bins=[x_edges, y_edges])[0] > 0
    pred_grid = np.histogram2d(pred_xy[:, 0], pred_xy[:, 1], bins=[x_edges, y_edges])[0] > 0
    intersection = np.logical_and(real_grid, pred_grid).sum()
    union_count = np.logical_or(real_grid, pred_grid).sum()
    real_std = real_xy.std(axis=0)
    pred_std = pred_xy.std(axis=0)
    categories = sorted(set(real.obs[annotation_key].astype(str)) & set(pred.obs[annotation_key].astype(str)))
    centroid_distances = []
    for category in categories:
        real_mask = real.obs[annotation_key].astype(str).to_numpy() == category
        pred_mask = pred.obs[annotation_key].astype(str).to_numpy() == category
        if real_mask.any() and pred_mask.any():
            centroid_distances.append(float(np.linalg.norm(real_xy[real_mask].mean(axis=0) - pred_xy[pred_mask].mean(axis=0))))
    return {
        "spatial_chamfer_norm": float(chamfer / diag if diag > 0 else np.nan),
        "spatial_grid_iou": float(intersection / union_count if union_count else np.nan),
        "centroid_mean": float(np.mean(centroid_distances)) if centroid_distances else np.nan,
        "std_ratio_x": float(pred_std[0] / real_std[0]),
        "std_ratio_y": float(pred_std[1] / real_std[1]),
    }


def _plot_real_pred(real: ad.AnnData, pred: ad.AnnData, path: Path, title: str, annotation_key: str, spatial_key: str) -> None:
    cmap = _color_map(real, annotation_key)
    categories = list(real.obs[annotation_key].cat.categories)
    real_xy = np.asarray(real.obsm[spatial_key], dtype=np.float32)
    pred_xy = np.asarray(pred.obsm[spatial_key], dtype=np.float32)
    union = np.vstack([real_xy, pred_xy])
    xpad = 0.04 * (union[:, 0].max() - union[:, 0].min())
    ypad = 0.04 * (union[:, 1].max() - union[:, 1].min())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, panel, label in [(axes[0], real, "real"), (axes[1], pred, "predicted")]:
        xy = np.asarray(panel.obsm[spatial_key], dtype=np.float32)
        labels = panel.obs[annotation_key].astype(str).to_numpy()
        for category in categories:
            mask = labels == category
            if mask.any():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=7.0, c=[cmap.get(category, "#999999")], linewidths=0, alpha=1.0)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(union[:, 0].min() - xpad, union[:, 0].max() + xpad)
        ax.set_ylim(union[:, 1].min() - ypad, union[:, 1].max() + ypad)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _predict_state(
    model: STCTDModel,
    data: PreparedData,
    train_config: TrainConfig,
    target_index: int,
    *,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state0 = data.state_by_time[0]
    if target_index == 0:
        state_np = state0.detach().cpu().numpy().astype(np.float32)
        weights = np.ones(state0.shape[0], dtype=np.float32)
    else:
        state_chunks = []
        weight_chunks = []
        for start in range(0, state0.shape[0], chunk_size):
            stop = min(start + chunk_size, state0.shape[0])
            state0_chunk = state0[start:stop]
            logw0 = torch.zeros(state0_chunk.shape[0], 1, dtype=state0_chunk.dtype, device=state0_chunk.device)
            if str(getattr(train_config, "loss_mode", "")).lower() == "stctd_fullgrid":
                steps = max(1, int(math.ceil(abs(float(data.time_values[target_index] - data.time_values[0])) / float(train_config.ode_step_size))))
            else:
                steps = max(1, train_config.steps_per_interval * target_index)
            state, logw, _, _ = rollout_stctd(
                model,
                state0_chunk,
                logw0,
                data.time_values[0],
                data.time_values[target_index],
                steps=steps,
                method=train_config.integrator,
                alpha_exp=train_config.alpha_exp,
                alpha_gro=train_config.alpha_gro if train_config.use_growth else 0.0,
            )
            state_chunks.append(state.detach().cpu().numpy().astype(np.float32))
            if train_config.use_growth:
                chunk_weights = torch.exp(logw).reshape(-1)
            else:
                chunk_weights = torch.ones(logw.shape[0], dtype=logw.dtype, device=logw.device)
            weight_chunks.append(chunk_weights.detach().cpu().numpy().astype(np.float32))
            del state, logw, logw0, state0_chunk
            if state0.device.type in {"mps", "cuda"}:
                torch.mps.empty_cache() if state0.device.type == "mps" else torch.cuda.empty_cache()
        state_np = np.vstack(state_chunks).astype(np.float32)
        weights = np.concatenate(weight_chunks).astype(np.float32)
    weights = weights.astype(np.float64)
    weights = np.maximum(weights, 1e-8)
    probs = weights / weights.sum()
    n_target = data.state_by_time[target_index].shape[0]
    rng = np.random.default_rng(train_config.seed + target_index)
    chosen = np.arange(state_np.shape[0]) if state_np.shape[0] == n_target else rng.choice(state_np.shape[0], size=n_target, replace=True, p=probs)
    return state_np, weights.astype(np.float32), chosen.astype(int)


@torch.no_grad()
def _spatiotemporal_labels(
    classifier: torch.nn.Module,
    label_to_cell_type_map: dict[int, str],
    spatial: np.ndarray,
    latent: np.ndarray,
    time_value: float,
    device: torch.device,
) -> np.ndarray:
    classifier.eval()
    labels = []
    for start in range(0, spatial.shape[0], 4096):
        stop = min(start + 4096, spatial.shape[0])
        batch_spatial = torch.as_tensor(spatial[start:stop], dtype=torch.float32, device=device)
        batch_latent = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
        batch_time = torch.full((batch_spatial.shape[0], 1), float(time_value), dtype=torch.float32, device=device)
        logits = classifier(torch.cat((batch_spatial, batch_latent, batch_time), dim=1))
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        labels.extend(str(label_to_cell_type_map[int(idx)]) for idx in pred)
    return np.asarray(labels, dtype=object)


def _train_stctd_spatiotemporal_classifier(
    data: PreparedData,
    data_config: DataConfig,
    output_dir: Path,
) -> tuple[torch.nn.Module, dict[int, str], Path]:
    observed = data.adata.copy()
    observed.obs[data_config.annotation_key] = observed.obs[data_config.annotation_key].astype("category")
    classifier_path = output_dir / "spatiotemporal_classifier.pt"
    label_to_cell_type_map = create_spatiotemporal_classifier(
        observed,
        str(classifier_path),
        annotation_key=data_config.annotation_key,
        device=data.state_by_time[0].device,
        spatial_key=data_config.spatial_key,
        latent_key=data_config.latent_key,
        time_key=data_config.time_key,
    )
    (output_dir / "label_to_cell_type_map.json").write_text(
        json.dumps({str(k): str(v) for k, v in label_to_cell_type_map.items()}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    classifier = torch.load(classifier_path, map_location=data.state_by_time[0].device, weights_only=False)
    return classifier.to(data.state_by_time[0].device).eval(), {int(k): str(v) for k, v in label_to_cell_type_map.items()}, classifier_path


def _build_pred_adata(
    data: PreparedData,
    data_config: DataConfig,
    decoder: DecoderRuntime | None,
    latent_classifier: TrainedLatentClassifier | None,
    st_classifier: torch.nn.Module | None,
    st_label_map: dict[int, str] | None,
    pred_state_all: np.ndarray,
    pred_weights_all: np.ndarray,
    chosen: np.ndarray,
    target_index: int,
    label_source: str,
) -> ad.AnnData:
    pred_state = pred_state_all[chosen]
    spatial, latent = data.scaler.inverse(pred_state)
    if label_source == "spatiotemporal_classifier":
        if st_classifier is None or st_label_map is None:
            raise ValueError("spatiotemporal label source requires a trained classifier and label map.")
        labels = _spatiotemporal_labels(
            st_classifier,
            st_label_map,
            spatial,
            latent,
            data.time_values[target_index],
            data.state_by_time[0].device,
        )
    else:
        if latent_classifier is None:
            raise ValueError("latent_z_classifier label source requires a trained latent classifier.")
        labels = predict_latent_labels(latent_classifier, latent, device=data.state_by_time[0].device)
    if decoder is None:
        decoded = sp.csr_matrix((pred_state.shape[0], data.expression_dim), dtype=np.float32)
    else:
        decoded_chunks = []
        decode_chunk = 1024
        with torch.enable_grad():
            for start in range(0, pred_state.shape[0], decode_chunk):
                stop = min(start + decode_chunk, pred_state.shape[0])
                pred_state_t = torch.as_tensor(pred_state[start:stop], dtype=torch.float32, device=data.state_by_time[0].device)
                decoded_chunks.append(_decode_pred_expression(pred_state_t, data, decoder).detach().cpu().numpy().astype(np.float32))
                del pred_state_t
                device = data.state_by_time[0].device
                if device.type in {"mps", "cuda"}:
                    torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()
        decoded = np.vstack(decoded_chunks).astype(np.float32)
    pred = ad.AnnData(X=sp.csr_matrix(decoded), var=data.adata.var.copy())
    pred.obs_names = [f"stctd_pred_{data.raw_time_values[target_index]:g}_{i}" for i in range(pred.n_obs)]
    pred.obs[data_config.annotation_key] = pd.Categorical(labels, categories=data.adata.obs[data_config.annotation_key].cat.categories)
    pred.obs[data_config.raw_time_key] = float(data.raw_time_values[target_index])
    pred.obs[data_config.time_key] = float(data.time_values[target_index])
    pred.obs["label_source"] = str(label_source)
    pred.obs["source_initial_index"] = chosen
    pred.obs["source_weight"] = pred_weights_all[chosen]
    pred.obsm[data_config.spatial_key] = spatial
    pred.obsm[data_config.latent_key] = latent
    pred.uns["stctd_expression_source"] = "decoder" if decoder is not None else "zeros_no_decoder"
    pred.uns["Annotation_colors"] = data.adata.uns.get("Annotation_colors", [])
    pred.uns[f"{data_config.annotation_key}_colors"] = data.adata.uns.get(f"{data_config.annotation_key}_colors", pred.uns["Annotation_colors"])
    return pred


def evaluate_stctd_model(
    model: STCTDModel,
    data: PreparedData,
    decoder_checkpoint: Path | None,
    train_config: TrainConfig,
    data_config: DataConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    comparison_dir = output_dir / "comparisons"
    pred_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    decoder = load_decoder_runtime(decoder_checkpoint, data.state_by_time[0].device) if decoder_checkpoint is not None else None
    label_source = "spatiotemporal_classifier" if str(getattr(train_config, "loss_mode", "")).lower() == "stctd_fullgrid" else "latent_z_classifier"
    latent_classifier = None
    st_classifier = None
    st_label_map = None
    st_classifier_path = None
    if label_source == "spatiotemporal_classifier":
        st_classifier, st_label_map, st_classifier_path = _train_stctd_spatiotemporal_classifier(data, data_config, output_dir)
    else:
        latent_classifier = train_latent_classifier(
            data,
            data_config.annotation_key,
            output_dir=output_dir / "latent_classifier",
            config=LatentClassifierConfig(epochs=500, patience=60, seed=train_config.seed),
        )
    observed = data.adata.copy()
    observed.obs[data_config.annotation_key] = observed.obs[data_config.annotation_key].astype("category")
    metrics = []
    mass_rows = []
    model.eval()
    for target_index, raw_time in enumerate(data.raw_time_values):
        state_all, weights_all, chosen = _predict_state(model, data, train_config, target_index)
        pred = _build_pred_adata(
            data,
            data_config,
            decoder,
            latent_classifier,
            st_classifier,
            st_label_map,
            state_all,
            weights_all,
            chosen,
            target_index,
            label_source,
        )
        suffix = str(raw_time).replace(".", "p")
        pred_path = pred_dir / f"predict_{suffix}.h5ad"
        pred.write_h5ad(pred_path)
        real = observed[np.isclose(observed.obs[data_config.raw_time_key].astype(float), raw_time)].copy()
        real.obs[data_config.annotation_key] = real.obs[data_config.annotation_key].astype("category")
        row: dict[str, Any] = {"time": float(raw_time), "time_input": float(data.time_values[target_index]), "pred_h5ad": str(pred_path)}
        row.update(_label_metrics(real, pred, data_config.annotation_key))
        row.update(_spatial_metrics(real, pred, data_config.annotation_key, data_config.spatial_key))
        plot_path = comparison_dir / f"{suffix}_real_vs_pred.png"
        _plot_real_pred(real, pred, plot_path, f"stCTD E{raw_time:g}", data_config.annotation_key, data_config.spatial_key)
        row["compare_png"] = str(plot_path)
        metrics.append(row)
        expected_ratio = data.state_by_time[target_index].shape[0] / data.state_by_time[0].shape[0]
        mass_rows.append(
            {
                "time": float(raw_time),
                "expected_ratio_from_initial": float(expected_ratio),
                "pred_mean_weight": float(weights_all.mean()),
                "pred_sum_weight": float(weights_all.sum()),
                "n_initial": int(data.state_by_time[0].shape[0]),
                "n_target": int(data.state_by_time[target_index].shape[0]),
            }
        )
    metrics_csv = output_dir / "metrics.csv"
    mass_csv = output_dir / "mass_diagnostics.csv"
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
    pd.DataFrame(mass_rows).to_csv(mass_csv, index=False)
    return {
        "metrics_csv": str(metrics_csv),
        "mass_diagnostics_csv": str(mass_csv),
        "predictions_dir": str(pred_dir),
        "comparisons_dir": str(comparison_dir),
        "label_source": label_source,
        "expression_source": "decoder" if decoder is not None else "zeros_no_decoder",
        "prediction_cell_counts": "resampled_to_observed_counts",
        "latent_classifier": {
            "checkpoint": str(latent_classifier.checkpoint_path),
            "trace": str(latent_classifier.trace_path),
            "config": latent_classifier.config.to_json_dict(),
            "categories": latent_classifier.categories,
        }
        if latent_classifier is not None
        else None,
        "spatiotemporal_classifier": {
            "checkpoint": str(st_classifier_path),
            "label_to_cell_type_map": str(output_dir / "label_to_cell_type_map.json"),
        }
        if st_classifier_path is not None
        else None,
        "final": metrics[-1],
        "metrics": metrics,
        "mass_diagnostics": mass_rows,
    }

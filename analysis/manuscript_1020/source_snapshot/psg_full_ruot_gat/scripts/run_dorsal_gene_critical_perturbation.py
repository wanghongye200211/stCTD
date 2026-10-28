#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy import stats
from torchdiffeq import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STVCR_SRC = PROJECT_ROOT / "stVCR_repo" / "src"
JOINTAE_SRC = PROJECT_ROOT / "stvcr_jointae" / "src"
SPAPOT_SRC = PROJECT_ROOT / "SpaPOT" / "src"
for path in (STVCR_SRC, JOINTAE_SRC, SPAPOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stvcr.training.model import stVCR_DynamicModel  # noqa: E402
try:
    from stvcr_jointae.preprocessing.gene_prior_gatae import (  # type: ignore # noqa: E402
        GenePriorGATConfig,
        GenePriorGATAutoEncoder,
    )
except ModuleNotFoundError:
    from embedding.preprocessing.gene_prior_gatae import (  # noqa: E402
        GenePriorGATConfig,
        GenePriorGATAutoEncoder,
    )


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "SpaPOT_runs"
    / "dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "result_figures"
    / "dorsal_midbrain_12p5_14p5_16p5_hybrid"
    / "gene_perturbation_regvelo_style_v1"
)
DEFAULT_RAW_DATA = PROJECT_ROOT / "data" / "mouse_dorsal_midbrain" / "Dorsal_midbrain_cell_bin.h5ad"
DEFAULT_MARKERS = [
    "Mki67",
    "Dbx1",
    "Npas3",
    "Adgrb3",
    "Aldh1l1",
    "Efna5",
    "Sox2",
    "Pax6",
    "Ascl1",
    "Neurog2",
    "Neurod1",
    "Gfap",
    "Olig2",
]
FORCED_GENES = ["Dbx1", "Adgrb3", "Neurod1", "Efna5"]
CELL_TYPES = ["RGC", "NeuB", "GlioB"]
COLORS = {"RGC": "#2ca25f", "NeuB": "#f28e2b", "GlioB": "#4e79a7"}


@dataclass
class GATBundle:
    model: GenePriorGATAutoEncoder
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    gene_names: list[str]
    checkpoint_path: Path
    dynamics_to_decoder_matrix: torch.Tensor | None = None
    dynamics_to_decoder_bias: torch.Tensor | None = None
    decoder_to_dynamics_matrix: torch.Tensor | None = None
    decoder_to_dynamics_bias: torch.Tensor | None = None
    alignment_metrics: dict[str, Any] | None = None


@dataclass
class KNNReadout:
    x_scaled: np.ndarray
    labels: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    k: int


@dataclass
class ControlPack:
    source_state: torch.Tensor
    source_logw: torch.Tensor
    decoded_expression: torch.Tensor
    control_state: torch.Tensor
    control_rollout: dict[str, torch.Tensor]
    control_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dorsal midbrain gene-level in silico perturbation screen. "
            "Perturbation can be introduced as a mild z+x latent shift whose gene-specific direction "
            "is estimated by the trained GAT encoder, then rolled out by the trained hybrid/stVCR dynamics."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-data", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=None,
        help="Optional GAT-AE checkpoint override. If it contains stvCR latent alignment metadata, it is used for encode/decode.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--source-times", nargs="*", type=float, default=[13.0, 13.6, 14.0])
    parser.add_argument("--critical-time", type=float, default=13.6)
    parser.add_argument("--terminal-time", type=float, default=16.5)
    parser.add_argument("--source-label", default="RGC")
    parser.add_argument("--screen-doses", nargs="*", type=float, default=[-2.0, 2.0])
    parser.add_argument("--dose-response-doses", nargs="*", type=float, default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    parser.add_argument(
        "--perturbation-mode",
        choices=["latent_add", "expression_encode"],
        default="latent_add",
        help=(
            "latent_add uses z + x along the GAT-induced gene direction. "
            "expression_encode replaces z by the full GAT encoding of the perturbed expression."
        ),
    )
    parser.add_argument(
        "--latent-add-x",
        type=float,
        default=0.32,
        help="Latent L2 shift used by latent_add when |dose|=2. This is a moderate fate-response perturbation.",
    )
    parser.add_argument(
        "--latent-add-max-x",
        type=float,
        default=0.64,
        help="Maximum latent L2 shift allowed by latent_add.",
    )
    parser.add_argument(
        "--latent-direction-expression-dose",
        type=float,
        default=1.0,
        help="Small expression z-score step used only to estimate the gene-specific latent direction.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=19491001)
    parser.add_argument("--max-source-cells", type=int, default=400)
    parser.add_argument("--prescreen-cells", type=int, default=192)
    parser.add_argument("--top-screen-genes", type=int, default=50)
    parser.add_argument("--top-dose-genes", type=int, default=10)
    parser.add_argument("--genes", nargs="*", default=None, help="Optional explicit genes. If set, skips top-gene selection.")
    parser.add_argument("--smoke", action="store_true", help="Run the forced marker smoke test only.")
    parser.add_argument("--metric-dt", type=float, default=0.1)
    parser.add_argument("--ot-sample-size", type=int, default=256)
    parser.add_argument("--sinkhorn-iters", type=int, default=60)
    parser.add_argument("--sinkhorn-epsilon-scale", type=float, default=0.08)
    parser.add_argument("--ode-chunk-size", type=int, default=512)
    parser.add_argument("--gat-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=4096)
    parser.add_argument("--knn-k", type=int, default=31)
    parser.add_argument("--expression-clip-quantile", type=float, default=0.999)
    parser.add_argument("--gene-input-key", default="X_gene_input")
    parser.add_argument("--spatial-key", default="X_spatial_aligned")
    parser.add_argument("--time-input-key", default="time_input")
    parser.add_argument("--raw-time-key", default="time")
    parser.add_argument("--annotation-key", default="Annotation")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if name in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if name in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dense_float32(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def observed_time_pairs(adata: ad.AnnData, args: argparse.Namespace) -> pd.DataFrame:
    obs = pd.DataFrame(
        {
            "time_input": np.asarray(adata.obs[args.time_input_key], dtype=np.float64),
            "raw_time": np.asarray(adata.obs[args.raw_time_key], dtype=np.float64),
        }
    )
    return obs.groupby("time_input", as_index=False)["raw_time"].median().sort_values("time_input")


def raw_to_input(raw_time: float, pairs: pd.DataFrame) -> float:
    return float(np.interp(float(raw_time), pairs["raw_time"].to_numpy(), pairs["time_input"].to_numpy()))


def input_to_raw(input_time: float, pairs: pd.DataFrame) -> float:
    return float(np.interp(float(input_time), pairs["time_input"].to_numpy(), pairs["raw_time"].to_numpy()))


def load_dynamics(summary: dict[str, Any], adata: ad.AnnData, args: argparse.Namespace, device: torch.device) -> stVCR_DynamicModel:
    payload = torch.load(summary["model_path"], map_location=device, weights_only=False)
    if isinstance(payload, stVCR_DynamicModel):
        return payload.to(device).eval()
    cfg = dict(summary.get("config", {}))
    spatial_dim = int(adata.obsm[args.spatial_key].shape[1])
    gene_dim = int(adata.obsm[args.gene_input_key].shape[1])
    model = stVCR_DynamicModel(
        in_out_gene_dim=gene_dim,
        spatial_dim=spatial_dim,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        n_hiddens=int(cfg.get("n_hiddens", 6)),
        activation=str(cfg.get("activation", "relu")),
        use_gene=True,
        use_spatial=True,
        velocity_parameterization=str(cfg.get("velocity_parameterization", "hybrid")),
    ).to(device)
    model.load_state_dict(payload)
    return model.eval()


def load_classifier(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[int, str], Path]:
    classifier_path = run_dir / "spatiotemporal_classifier.pt"
    label_map_path = run_dir / "label_to_cell_type_map.json"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Missing classifier: {classifier_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_map_path}")
    label_map = {int(k): str(v) for k, v in load_json(label_map_path).items()}
    classifier = torch.load(classifier_path, map_location=device, weights_only=False).to(device).eval()
    return classifier, label_map, classifier_path


def _alignment_tensor(
    alignment: dict[str, Any],
    key: str,
    device: torch.device,
) -> torch.Tensor | None:
    value = alignment.get(key)
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def load_gat_bundle(
    adata: ad.AnnData,
    device: torch.device,
    checkpoint_override: Path | None = None,
) -> GATBundle:
    ready = dict(adata.uns.get("gene_prior_gatae_stvcr_ready", {}))
    info = dict(adata.uns.get("gene_prior_gatae", {}))
    checkpoint_path = Path(
        checkpoint_override
        or ready.get("decoder_checkpoint_path")
        or info.get("checkpoint_path")
        or ""
    ).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing GAT checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "gene_prior_gatae":
        raise ValueError(f"{checkpoint_path} is not a gene_prior_gatae checkpoint.")
    config = GenePriorGATConfig(**checkpoint["config"])
    edge_index = checkpoint["edge_index"].to(device)
    model = GenePriorGATAutoEncoder(len(checkpoint["gene_names"]), edge_index, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    latent_mean = torch.as_tensor(checkpoint["latent_mean"], dtype=torch.float32, device=device)
    latent_std = torch.as_tensor(checkpoint["latent_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    alignment = checkpoint.get("stvcr_latent_alignment")
    if isinstance(alignment, dict):
        dynamics_to_decoder_matrix = _alignment_tensor(alignment, "old_to_new_matrix", device)
        dynamics_to_decoder_bias = _alignment_tensor(alignment, "old_to_new_bias", device)
        decoder_to_dynamics_matrix = _alignment_tensor(alignment, "new_to_old_matrix", device)
        decoder_to_dynamics_bias = _alignment_tensor(alignment, "new_to_old_bias", device)
        alignment_metrics = {
            "old_to_new_metrics": alignment.get("old_to_new_metrics", {}),
            "new_to_old_metrics": alignment.get("new_to_old_metrics", {}),
            "n_shared_cells": alignment.get("n_shared_cells"),
        }
    else:
        dynamics_to_decoder_matrix = None
        dynamics_to_decoder_bias = None
        decoder_to_dynamics_matrix = None
        decoder_to_dynamics_bias = None
        alignment_metrics = None
    return GATBundle(
        model=model,
        latent_mean=latent_mean,
        latent_std=latent_std,
        gene_names=list(map(str, checkpoint["gene_names"])),
        checkpoint_path=checkpoint_path.resolve(),
        dynamics_to_decoder_matrix=dynamics_to_decoder_matrix,
        dynamics_to_decoder_bias=dynamics_to_decoder_bias,
        decoder_to_dynamics_matrix=decoder_to_dynamics_matrix,
        decoder_to_dynamics_bias=decoder_to_dynamics_bias,
        alignment_metrics=alignment_metrics,
    )


def standardize_latent(bundle: GATBundle, latent_raw: torch.Tensor) -> torch.Tensor:
    return (latent_raw - bundle.latent_mean.reshape(1, -1)) / bundle.latent_std.reshape(1, -1)


def unstandardize_latent(bundle: GATBundle, latent_std: torch.Tensor) -> torch.Tensor:
    return latent_std * bundle.latent_std.reshape(1, -1) + bundle.latent_mean.reshape(1, -1)


def _apply_affine(
    latent_std: torch.Tensor,
    matrix: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if matrix is None or bias is None:
        return latent_std
    return latent_std @ matrix + bias.reshape(1, -1)


def dynamics_to_decoder_standardized(bundle: GATBundle, latent_std: torch.Tensor) -> torch.Tensor:
    return _apply_affine(latent_std, bundle.dynamics_to_decoder_matrix, bundle.dynamics_to_decoder_bias)


def decoder_to_dynamics_standardized(bundle: GATBundle, latent_std: torch.Tensor) -> torch.Tensor:
    return _apply_affine(latent_std, bundle.decoder_to_dynamics_matrix, bundle.decoder_to_dynamics_bias)


@torch.no_grad()
def decode_standardized_latent(
    bundle: GATBundle,
    latent_std: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, latent_std.shape[0], int(batch_size)):
        batch = latent_std[start : start + int(batch_size)]
        decoder_batch = dynamics_to_decoder_standardized(bundle, batch)
        raw = unstandardize_latent(bundle, decoder_batch)
        outputs.append(bundle.model.decoder(raw).detach())
        clear_device_cache(device)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_expression(
    bundle: GATBundle,
    expression: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    for start in range(0, expression.shape[0], int(batch_size)):
        batch = expression[start : start + int(batch_size)]
        latent_raw = bundle.model.encoder(batch)
        decoder_std = standardize_latent(bundle, latent_raw)
        latents.append(decoder_to_dynamics_standardized(bundle, decoder_std).detach())
        clear_device_cache(device)
    return torch.cat(latents, dim=0)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def integrate_model(
    model: stVCR_DynamicModel,
    state: torch.Tensor,
    logw: torch.Tensor,
    t0: float,
    t1: float,
    *,
    method: str,
    step_size: float,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if abs(float(t1) - float(t0)) < 1e-12:
        return state.detach(), logw.detach()
    states: list[torch.Tensor] = []
    logws: list[torch.Tensor] = []
    t = torch.tensor([float(t0), float(t1)], dtype=torch.float32, device=device)
    options = {"step_size": float(step_size)} if method in {"rk4", "euler", "midpoint"} else {}
    for start in range(0, state.shape[0], int(chunk_size)):
        chunk = state[start : start + int(chunk_size)].detach()
        chunk_logw = logw[start : start + int(chunk_size)].detach()
        with torch.enable_grad():
            state_t, logw_t = odeint(
                model,
                (chunk, chunk_logw),
                t=t,
                atol=1e-5,
                rtol=1e-5,
                method=method,
                options=options,
            )
        states.append(state_t[-1].detach())
        logws.append(logw_t[-1].detach())
        clear_device_cache(device)
    return torch.cat(states, dim=0), torch.cat(logws, dim=0)


def dynamics_vector(
    model: stVCR_DynamicModel,
    state: torch.Tensor,
    logw: torch.Tensor,
    t_value: float,
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dz_rows: list[torch.Tensor] = []
    dlogw_rows: list[torch.Tensor] = []
    t = torch.tensor(float(t_value), dtype=torch.float32, device=device)
    for start in range(0, state.shape[0], int(chunk_size)):
        chunk = state[start : start + int(chunk_size)].detach().clone()
        chunk_logw = logw[start : start + int(chunk_size)].detach().clone()
        with torch.enable_grad():
            dz_dt, dlogw_dt = model(t, (chunk, chunk_logw))
        dz_rows.append(dz_dt.detach())
        dlogw_rows.append(dlogw_dt.detach())
        clear_device_cache(device)
    return torch.cat(dz_rows, dim=0), torch.cat(dlogw_rows, dim=0)


def vector_effect_stats(
    base: torch.Tensor,
    pert: torch.Tensor,
    spatial_dim: int,
) -> dict[str, np.ndarray | float]:
    base_s = base[:, :spatial_dim]
    pert_s = pert[:, :spatial_dim]
    base_z = base[:, spatial_dim:]
    pert_z = pert[:, spatial_dim:]
    effect_s = (1.0 - torch.nn.functional.cosine_similarity(base_s, pert_s, dim=1, eps=1e-8)).detach().cpu().numpy()
    effect_z = (1.0 - torch.nn.functional.cosine_similarity(base_z, pert_z, dim=1, eps=1e-8)).detach().cpu().numpy()
    effect_joint = (1.0 - torch.nn.functional.cosine_similarity(base, pert, dim=1, eps=1e-8)).detach().cpu().numpy()
    delta_s = torch.linalg.norm(pert_s - base_s, dim=1).detach().cpu().numpy()
    delta_z = torch.linalg.norm(pert_z - base_z, dim=1).detach().cpu().numpy()
    return {
        "local_effect_spatial": effect_s,
        "local_effect_gene": effect_z,
        "local_effect_joint": effect_joint,
        "local_delta_velocity_spatial_norm": delta_s,
        "local_delta_velocity_gene_norm": delta_z,
        "local_effect_spatial_mean": float(np.nanmean(effect_s)),
        "local_effect_spatial_median": float(np.nanmedian(effect_s)),
        "local_effect_spatial_q90": float(np.nanquantile(effect_s, 0.90)),
        "local_effect_gene_mean": float(np.nanmean(effect_z)),
        "local_effect_gene_median": float(np.nanmedian(effect_z)),
        "local_effect_gene_q90": float(np.nanquantile(effect_z, 0.90)),
        "local_effect_joint_mean": float(np.nanmean(effect_joint)),
        "local_effect_joint_median": float(np.nanmedian(effect_joint)),
        "local_effect_joint_q90": float(np.nanquantile(effect_joint, 0.90)),
        "local_delta_velocity_spatial_norm_mean": float(np.nanmean(delta_s)),
        "local_delta_velocity_gene_norm_mean": float(np.nanmean(delta_z)),
    }


def potential_u(model: stVCR_DynamicModel, state: torch.Tensor, t_value: float) -> torch.Tensor:
    if not hasattr(model.gene_velocity_net, "potential"):
        return torch.full((state.shape[0],), float("nan"), dtype=state.dtype, device=state.device)
    with torch.enable_grad():
        state_in = state.detach().clone().requires_grad_(True)
        t = torch.tensor(float(t_value), dtype=state.dtype, device=state.device)
        u = model.gene_velocity_net.potential(t, state_in).reshape(-1)
    return u.detach()


def rollout_path(
    model: stVCR_DynamicModel,
    start_state: torch.Tensor,
    start_logw: torch.Tensor,
    source_time: float,
    terminal_time: float,
    *,
    method: str,
    step_size: float,
    metric_dt: float,
    chunk_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n_steps = max(1, int(math.ceil((float(terminal_time) - float(source_time)) / max(float(metric_dt), 1e-8))))
    times = np.linspace(float(source_time), float(terminal_time), n_steps + 1)
    state = start_state.detach()
    logw = start_logw.detach()
    prev_u = potential_u(model, state, float(times[0]))
    integrated_abs = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
    for idx in range(1, len(times)):
        state, logw = integrate_model(
            model,
            state,
            logw,
            float(times[idx - 1]),
            float(times[idx]),
            method=method,
            step_size=step_size,
            chunk_size=chunk_size,
            device=device,
        )
        cur_u = potential_u(model, state, float(times[idx]))
        integrated_abs = integrated_abs + torch.abs(cur_u - prev_u)
        prev_u = cur_u
    return {
        "state": state.detach(),
        "logw": logw.detach(),
        "terminal_u": prev_u.detach(),
        "integrated_abs_step_dU": integrated_abs.detach(),
    }


@torch.no_grad()
def classifier_readout(
    classifier: torch.nn.Module,
    label_map: dict[int, str],
    state: torch.Tensor,
    time_value: float,
    spatial_dim: int,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = [label_map[idx] for idx in sorted(label_map)]
    probs: list[np.ndarray] = []
    labels: list[str] = []
    for start in range(0, state.shape[0], int(batch_size)):
        batch = state[start : start + int(batch_size)]
        t = torch.full((batch.shape[0], 1), float(time_value), dtype=batch.dtype, device=batch.device)
        logits = classifier(torch.cat((batch[:, :spatial_dim], batch[:, spatial_dim:], t), dim=1))
        p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(p)
        labels.extend(classes[int(i)] for i in p.argmax(axis=1))
    return np.vstack(probs), np.asarray(labels, dtype=object), classes


def build_terminal_knn(
    adata: ad.AnnData,
    args: argparse.Namespace,
    terminal_input: float,
    spatial_dim: int,
    k: int,
) -> KNNReadout:
    time_input = np.asarray(adata.obs[args.time_input_key], dtype=np.float64)
    mask = np.isclose(time_input, terminal_input, atol=1e-4)
    state = np.concatenate(
        [
            np.asarray(adata.obsm[args.spatial_key][mask], dtype=np.float32),
            np.asarray(adata.obsm[args.gene_input_key][mask], dtype=np.float32),
        ],
        axis=1,
    )
    t = np.full((state.shape[0], 1), float(terminal_input), dtype=np.float32)
    x = np.concatenate([state[:, :spatial_dim], state[:, spatial_dim:], t], axis=1)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return KNNReadout(
        x_scaled=((x - mean) / std).astype(np.float32),
        labels=adata.obs.loc[mask, args.annotation_key].astype(str).to_numpy(),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        k=int(min(k, max(1, state.shape[0]))),
    )


def knn_readout(knn: KNNReadout, state: np.ndarray, terminal_input: float, spatial_dim: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = CELL_TYPES
    t = np.full((state.shape[0], 1), float(terminal_input), dtype=np.float32)
    x = np.concatenate([state[:, :spatial_dim], state[:, spatial_dim:], t], axis=1)
    x_scaled = ((x - knn.mean) / knn.std).astype(np.float32)
    probs = np.zeros((state.shape[0], len(classes)), dtype=np.float32)
    hard: list[str] = []
    for start in range(0, x_scaled.shape[0], 1024):
        q = x_scaled[start : start + 1024]
        dist = ((q[:, None, :] - knn.x_scaled[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(dist, kth=knn.k - 1, axis=1)[:, : knn.k]
        neighbor_labels = knn.labels[idx]
        for row_idx, labels in enumerate(neighbor_labels):
            counts = np.asarray([(labels == label).sum() for label in classes], dtype=np.float32)
            probs[start + row_idx] = counts / max(float(knn.k), 1.0)
            hard.append(classes[int(counts.argmax())])
    return probs, np.asarray(hard, dtype=object), classes


def sinkhorn_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sample_size: int,
    seed: int,
    epsilon_scale: float,
    n_iters: int,
) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    n = min(int(sample_size), int(x.shape[0]), int(y.shape[0]))
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    idx_x = torch.randperm(x.shape[0], device=x.device, generator=generator)[:n]
    idx_y = torch.randperm(y.shape[0], device=y.device, generator=generator)[:n]
    x_s = x[idx_x].float()
    y_s = y[idx_y].float()
    cost = torch.cdist(x_s, y_s).pow(2) / max(1, int(x_s.shape[1]))
    positive = cost[cost > 0]
    epsilon = torch.median(positive) * float(epsilon_scale) if positive.numel() else cost.new_tensor(1.0)
    epsilon = torch.clamp(epsilon, min=1e-4)
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=x.device)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32, device=x.device)
    log_a = torch.log(a.clamp_min(1e-12))
    log_b = torch.log(b.clamp_min(1e-12))
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(int(n_iters)):
        f = epsilon * (log_a - torch.logsumexp((g.reshape(1, -1) - cost) / epsilon, dim=1))
        g = epsilon * (log_b - torch.logsumexp((f.reshape(-1, 1) - cost) / epsilon, dim=0))
    plan = torch.exp((f.reshape(-1, 1) + g.reshape(1, -1) - cost) / epsilon)
    return float((plan * cost).sum().detach().cpu())


def expression_stats(expression: np.ndarray, clip_quantile: float) -> dict[str, np.ndarray]:
    std = expression.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    lo = np.zeros(expression.shape[1], dtype=np.float32)
    hi = np.quantile(expression, float(clip_quantile), axis=0).astype(np.float32)
    raw_hi = expression.max(axis=0).astype(np.float32)
    hi = np.maximum(hi, raw_hi * 0.5)
    hi = np.where(hi <= 0, raw_hi, hi)
    hi = np.where(hi <= 0, 1.0, hi)
    return {"std": std, "lo": lo, "hi": hi}


def select_source_indices(adata: ad.AnnData, args: argparse.Namespace, initial_input: float, seed: int) -> np.ndarray:
    time_input = np.asarray(adata.obs[args.time_input_key], dtype=np.float64)
    labels = adata.obs[args.annotation_key].astype(str).to_numpy()
    source = np.flatnonzero(np.isclose(time_input, initial_input, atol=1e-4) & (labels == str(args.source_label)))
    if int(args.max_source_cells) > 0 and source.shape[0] > int(args.max_source_cells):
        rng = np.random.default_rng(int(seed))
        source = np.sort(rng.choice(source, size=int(args.max_source_cells), replace=False))
    return source


def build_initial_state(adata: ad.AnnData, args: argparse.Namespace, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    state = np.concatenate(
        [
            np.asarray(adata.obsm[args.spatial_key][indices], dtype=np.float32),
            np.asarray(adata.obsm[args.gene_input_key][indices], dtype=np.float32),
        ],
        axis=1,
    )
    return torch.as_tensor(state, dtype=torch.float32, device=device)


def spatial_to_slice_coords(adata: ad.AnnData, args: argparse.Namespace, coords: np.ndarray) -> np.ndarray:
    """Map model spatial coordinates back to the raw-scale aligned ST slice when metadata is available."""
    coords = np.asarray(coords, dtype=np.float32)
    norm = adata.uns.get("hybrid_global_spatial_normalization", {})
    if isinstance(norm, dict) and "mean" in norm and "std" in norm:
        target_key = str(norm.get("target_spatial_key", ""))
        if target_key in {"X_spatial_input", args.spatial_key, "X_spatial_aligned"}:
            mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, -1)
            std = np.asarray(norm["std"], dtype=np.float32).reshape(1, -1)
            if mean.shape[1] == coords.shape[1] and std.shape[1] == coords.shape[1]:
                return coords * std + mean
    return coords


def observed_slice_coords(
    adata: ad.AnnData,
    args: argparse.Namespace,
    terminal_input: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_input = np.asarray(adata.obs[args.time_input_key], dtype=np.float64)
    terminal_mask = np.isclose(time_input, terminal_input, atol=1e-4)
    norm = adata.uns.get("hybrid_global_spatial_normalization", {})
    backup_key = norm.get("raw_spatial_backup_key") if isinstance(norm, dict) else None
    if backup_key in adata.obsm:
        observed = np.asarray(adata.obsm[backup_key][terminal_mask], dtype=np.float32)
    elif "X_spatial_input_raw_scale_preserved" in adata.obsm:
        observed = np.asarray(adata.obsm["X_spatial_input_raw_scale_preserved"][terminal_mask], dtype=np.float32)
    else:
        observed = spatial_to_slice_coords(adata, args, np.asarray(adata.obsm[args.spatial_key][terminal_mask], dtype=np.float32))
    labels = adata.obs[args.annotation_key].astype(str).to_numpy()[terminal_mask]
    return observed, labels


def make_control_pack(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    model: stVCR_DynamicModel,
    gat: GATBundle,
    classifier: torch.nn.Module,
    label_map: dict[int, str],
    knn: KNNReadout,
    source_input: float,
    terminal_input: float,
    source_indices: np.ndarray,
    initial_input: float,
    method: str,
    step_size: float,
    spatial_dim: int,
    expr_stats: dict[str, np.ndarray],
    device: torch.device,
) -> ControlPack:
    initial_state = build_initial_state(adata, args, source_indices, device)
    initial_logw = torch.zeros(initial_state.shape[0], 1, dtype=torch.float32, device=device)
    source_state, source_logw = integrate_model(
        model,
        initial_state,
        initial_logw,
        initial_input,
        source_input,
        method=method,
        step_size=step_size,
        chunk_size=int(args.ode_chunk_size),
        device=device,
    )
    decoded = decode_standardized_latent(
        gat,
        source_state[:, spatial_dim:],
        batch_size=int(args.gat_batch_size),
        device=device,
    )
    lo = torch.as_tensor(expr_stats["lo"], dtype=torch.float32, device=device).reshape(1, -1)
    hi = torch.as_tensor(expr_stats["hi"], dtype=torch.float32, device=device).reshape(1, -1)
    decoded = torch.minimum(torch.maximum(decoded, lo), hi)
    control_gene = encode_expression(gat, decoded, batch_size=int(args.gat_batch_size), device=device)
    control_state = source_state.detach().clone()
    control_state[:, spatial_dim:] = control_gene
    control_rollout = rollout_path(
        model,
        control_state,
        source_logw,
        source_input,
        terminal_input,
        method=method,
        step_size=step_size,
        metric_dt=float(args.metric_dt),
        chunk_size=int(args.ode_chunk_size),
        device=device,
    )
    control_metrics = summarize_rollout(
        rollout=control_rollout,
        reference_rollout=None,
        classifier=classifier,
        label_map=label_map,
        knn=knn,
        terminal_input=terminal_input,
        spatial_dim=spatial_dim,
        args=args,
        seed=0,
    )
    return ControlPack(
        source_state=source_state.detach(),
        source_logw=source_logw.detach(),
        decoded_expression=decoded.detach(),
        control_state=control_state.detach(),
        control_rollout=control_rollout,
        control_metrics=control_metrics,
    )


def summarize_rollout(
    *,
    rollout: dict[str, torch.Tensor],
    reference_rollout: dict[str, torch.Tensor] | None,
    classifier: torch.nn.Module,
    label_map: dict[int, str],
    knn: KNNReadout,
    terminal_input: float,
    spatial_dim: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    state = rollout["state"]
    probs, hard, classes = classifier_readout(
        classifier,
        label_map,
        state,
        terminal_input,
        spatial_dim,
        batch_size=int(args.classifier_batch_size),
    )
    state_np = state.detach().cpu().numpy().astype(np.float32)
    knn_probs, knn_hard, knn_classes = knn_readout(knn, state_np, terminal_input, spatial_dim)
    weights = torch.exp(rollout["logw"]).reshape(-1).detach().cpu().numpy()
    terminal_u = rollout["terminal_u"].detach().cpu().numpy()
    integrated = rollout["integrated_abs_step_dU"].detach().cpu().numpy()
    row: dict[str, Any] = {
        "n_cells": int(state.shape[0]),
        "terminal_U_mean": float(np.nanmean(terminal_u)),
        "terminal_U_median": float(np.nanmedian(terminal_u)),
        "integrated_abs_step_dU_mean": float(np.nanmean(integrated)),
        "integrated_abs_step_dU_median": float(np.nanmedian(integrated)),
        "terminal_mean_weight": float(np.mean(weights)),
        "terminal_sum_weight": float(np.sum(weights)),
        "classifier_knn_agreement": float(np.mean(hard == knn_hard)),
        "_classifier_hard": hard,
        "_classifier_probs": probs,
        "_classifier_classes": classes,
        "_knn_hard": knn_hard,
        "_knn_probs": knn_probs,
        "_knn_classes": knn_classes,
        "_state": state,
    }
    for label in CELL_TYPES:
        if label in classes:
            idx = classes.index(label)
            row[f"terminal_P_{label}"] = float(probs[:, idx].mean())
            row[f"terminal_frac_{label}"] = float(np.mean(hard == label))
            row[f"terminal_weight_{label}"] = float(weights[hard == label].sum())
        else:
            row[f"terminal_P_{label}"] = float("nan")
            row[f"terminal_frac_{label}"] = float("nan")
            row[f"terminal_weight_{label}"] = float("nan")
        if label in knn_classes:
            idx = knn_classes.index(label)
            row[f"knn_P_{label}"] = float(knn_probs[:, idx].mean())
            row[f"knn_frac_{label}"] = float(np.mean(knn_hard == label))
    if reference_rollout is None:
        row["terminal_OT_space"] = 0.0
        row["terminal_OT_gene"] = 0.0
    else:
        row["terminal_OT_space"] = sinkhorn_cost(
            reference_rollout["state"][:, :spatial_dim],
            state[:, :spatial_dim],
            sample_size=int(args.ot_sample_size),
            seed=int(seed),
            epsilon_scale=float(args.sinkhorn_epsilon_scale),
            n_iters=int(args.sinkhorn_iters),
        )
        row["terminal_OT_gene"] = sinkhorn_cost(
            reference_rollout["state"][:, spatial_dim:],
            state[:, spatial_dim:],
            sample_size=int(args.ot_sample_size),
            seed=int(seed),
            epsilon_scale=float(args.sinkhorn_epsilon_scale),
            n_iters=int(args.sinkhorn_iters),
        )
    return row


def public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def perturb_expression(
    expression: torch.Tensor,
    gene_idx: int,
    dose: float,
    expr_stats: dict[str, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    out = expression.detach().clone()
    std = float(expr_stats["std"][gene_idx])
    lo = float(expr_stats["lo"][gene_idx])
    hi = float(expr_stats["hi"][gene_idx])
    out[:, gene_idx] = torch.clamp(out[:, gene_idx] + float(dose) * std, min=lo, max=hi)
    return out


def latent_add_magnitude(dose: float, args: argparse.Namespace) -> float:
    if abs(float(dose)) < 1e-12:
        return 0.0
    scaled = abs(float(dose)) / 2.0 * float(args.latent_add_x)
    return float(min(scaled, float(args.latent_add_max_x)))


def gene_latent_add_shift(
    *,
    control: ControlPack,
    gene_idx: int,
    dose: float,
    gat: GATBundle,
    expr_stats: dict[str, np.ndarray],
    spatial_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    sign = 1.0 if float(dose) > 0 else -1.0
    direction_expr_dose = sign * float(args.latent_direction_expression_dose)
    direction_expr = perturb_expression(control.decoded_expression, gene_idx, direction_expr_dose, expr_stats, device)
    direction_gene = encode_expression(gat, direction_expr, batch_size=int(args.gat_batch_size), device=device)
    z0 = control.control_state[:, spatial_dim:]
    delta = direction_gene - z0
    delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
    unit = torch.where(delta_norm > 1e-8, delta / torch.clamp(delta_norm, min=1e-8), torch.zeros_like(delta))
    x_norm = latent_add_magnitude(float(dose), args)
    shift = float(x_norm) * unit
    info = {
        "latent_add_x": float(x_norm),
        "latent_direction_expression_dose": float(direction_expr_dose),
        "latent_direction_delta_norm_mean": float(delta_norm.detach().cpu().mean().item()),
        "latent_direction_delta_norm_median": float(delta_norm.detach().cpu().median().item()),
        "latent_shift_norm_mean": float(torch.linalg.norm(shift, dim=1).detach().cpu().mean().item()),
        "latent_shift_norm_median": float(torch.linalg.norm(shift, dim=1).detach().cpu().median().item()),
    }
    return z0 + shift, info


def row_for_condition(
    *,
    phase: str,
    gene: str,
    dose: float,
    seed: int,
    source_time: float,
    source_input: float,
    terminal_time: float,
    terminal_input: float,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "dataset": "dorsal_midbrain",
        "phase": phase,
        "source_time": float(source_time),
        "source_time_input": float(source_input),
        "terminal_time": float(terminal_time),
        "terminal_time_input": float(terminal_input),
        "gene": str(gene),
        "dose": float(dose),
        "seed": int(seed),
    }
    row.update(public_metrics(metrics))
    for label in CELL_TYPES:
        row[f"delta_P_{label}"] = float(metrics.get(f"terminal_P_{label}", np.nan) - baseline_metrics.get(f"terminal_P_{label}", np.nan))
        row[f"delta_frac_{label}"] = float(
            metrics.get(f"terminal_frac_{label}", np.nan) - baseline_metrics.get(f"terminal_frac_{label}", np.nan)
        )
        base_frac = float(baseline_metrics.get(f"terminal_frac_{label}", 0.0))
        frac = float(metrics.get(f"terminal_frac_{label}", 0.0))
        row[f"log2FC_{label}_fraction"] = float(np.log2((frac + 1e-3) / (base_frac + 1e-3)))
    base_hard = np.asarray(baseline_metrics["_classifier_hard"], dtype=object)
    hard = np.asarray(metrics["_classifier_hard"], dtype=object)
    rgc_base = base_hard == "RGC"
    denom = max(int(rgc_base.sum()), 1)
    row["switch_rate_RGC_to_NeuB"] = float(np.sum(rgc_base & (hard == "NeuB")) / denom)
    row["switch_rate_RGC_to_GlioB"] = float(np.sum(rgc_base & (hard == "GlioB")) / denom)
    row["any_classifier_label_change_rate"] = float(np.mean(hard != base_hard))
    row["delta_terminal_U_mean"] = float(metrics["terminal_U_mean"] - baseline_metrics["terminal_U_mean"])
    row["delta_terminal_sum_weight"] = float(metrics["terminal_sum_weight"] - baseline_metrics["terminal_sum_weight"])
    return row


def auc_raw_greater(raw: np.ndarray, pert: np.ndarray) -> float:
    raw = np.asarray(raw, dtype=np.float64)
    pert = np.asarray(pert, dtype=np.float64)
    n_raw = raw.shape[0]
    n_pert = pert.shape[0]
    if n_raw == 0 or n_pert == 0:
        return float("nan")
    scores = np.concatenate([raw, pert])
    labels = np.concatenate([np.ones(n_raw, dtype=bool), np.zeros(n_pert, dtype=bool)])
    ranks = stats.rankdata(scores)
    rank_sum_raw = ranks[labels].sum()
    return float((rank_sum_raw - n_raw * (n_raw + 1) / 2.0) / (n_raw * n_pert))


def p_adjust_bh(values: list[float]) -> list[float]:
    pvals = np.asarray(values, dtype=np.float64)
    out = np.full_like(pvals, np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)
    if not valid.any():
        return out.tolist()
    valid_p = pvals[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = ranked.shape[0]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    valid_indices = np.flatnonzero(valid)
    out[valid_indices[order]] = adjusted
    return out.tolist()


def fate_enrichment_rows(
    *,
    condition: dict[str, Any],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for readout, prob_key, class_key in (
        ("classifier", "_classifier_probs", "_classifier_classes"),
        ("knn", "_knn_probs", "_knn_classes"),
    ):
        base_probs = np.asarray(baseline_metrics.get(prob_key), dtype=np.float64)
        pert_probs = np.asarray(metrics.get(prob_key), dtype=np.float64)
        classes = list(metrics.get(class_key, []))
        if base_probs.ndim != 2 or pert_probs.ndim != 2 or not classes:
            continue
        for label in CELL_TYPES:
            if label not in classes:
                continue
            idx = classes.index(label)
            raw = base_probs[:, idx]
            pert = pert_probs[:, idx]
            try:
                p_two = float(stats.ranksums(pert, raw).pvalue)
                p_pert_greater = float(stats.ranksums(pert, raw, alternative="greater").pvalue)
                p_pert_less = float(stats.ranksums(pert, raw, alternative="less").pvalue)
            except Exception:
                p_two = p_pert_greater = p_pert_less = float("nan")
            delta = float(np.nanmean(pert) - np.nanmean(raw))
            rows.append(
                {
                    **condition,
                    "readout": readout,
                    "terminal_state": label,
                    "control_mean_probability": float(np.nanmean(raw)),
                    "perturbed_mean_probability": float(np.nanmean(pert)),
                    "delta_mean_probability": delta,
                    "regvelo_auc_raw_greater": auc_raw_greater(raw, pert),
                    "rank_sum_pvalue": p_two,
                    "rank_sum_pvalue_pert_greater": p_pert_greater,
                    "rank_sum_pvalue_pert_less": p_pert_less,
                    "effect_direction": "enrichment" if delta > 0 else ("depletion" if delta < 0 else "none"),
                }
            )
    return rows


def prescreen_genes(
    *,
    control: ControlPack,
    gat: GATBundle,
    classifier: torch.nn.Module,
    label_map: dict[int, str],
    expr_stats: dict[str, np.ndarray],
    source_input: float,
    spatial_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    n = min(int(args.prescreen_cells), int(control.decoded_expression.shape[0]))
    expr = control.decoded_expression[:n].detach().clone().requires_grad_(True)
    spatial = control.source_state[:n, :spatial_dim].detach()
    latent_raw = gat.model.encoder(expr)
    latent_std = decoder_to_dynamics_standardized(gat, standardize_latent(gat, latent_raw))
    state = torch.cat([spatial, latent_std], dim=1)
    t = torch.full((state.shape[0], 1), float(source_input), dtype=torch.float32, device=device)
    logits = classifier(torch.cat([state[:, :spatial_dim], state[:, spatial_dim:], t], dim=1))
    probs = torch.softmax(logits, dim=1)
    classes = [label_map[idx] for idx in sorted(label_map)]
    rows = []
    for target in ("NeuB", "GlioB"):
        if target not in classes:
            continue
        classifier.zero_grad(set_to_none=True)
        gat.model.zero_grad(set_to_none=True)
        target_score = probs[:, classes.index(target)].mean()
        grad = torch.autograd.grad(target_score, expr, retain_graph=True)[0]
        signed = grad.detach().mean(dim=0).cpu().numpy() * expr_stats["std"]
        abs_score = np.abs(grad.detach().cpu().numpy()).mean(axis=0) * expr_stats["std"]
        for idx, gene in enumerate(gat.gene_names):
            rows.append(
                {
                    "gene": gene,
                    "target": target,
                    "signed_gradient_score": float(signed[idx]),
                    "abs_gradient_score": float(abs_score[idx]),
                }
            )
    score = pd.DataFrame(rows)
    if score.empty:
        return pd.DataFrame({"gene": gat.gene_names, "combined_prescreen_score": 0.0})
    wide = score.pivot_table(index="gene", columns="target", values=["signed_gradient_score", "abs_gradient_score"], aggfunc="mean")
    out = pd.DataFrame({"gene": list(wide.index)})
    for metric in ("signed_gradient_score", "abs_gradient_score"):
        for target in ("NeuB", "GlioB"):
            key = (metric, target)
            out[f"{target}_{metric}"] = wide[key].to_numpy() if key in wide else 0.0
    out["combined_prescreen_score"] = out[[f"{target}_abs_gradient_score" for target in ("NeuB", "GlioB")]].max(axis=1)
    out = out.sort_values("combined_prescreen_score", ascending=False).reset_index(drop=True)
    return out


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    out = np.full_like(pvalues, np.nan)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return out
    p = pvalues[finite]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[finite] = tmp
    return out


def build_ranked_screen(summary_df: pd.DataFrame) -> pd.DataFrame:
    screen = summary_df[(summary_df["phase"] == "screen") & (summary_df["dose"] != 0)].copy()
    rows = []
    for (gene, source_time, dose), cur in screen.groupby(["gene", "source_time", "dose"]):
        row: dict[str, Any] = {"gene": gene, "source_time": float(source_time), "dose": float(dose), "n": int(len(cur))}
        for target in ("NeuB", "GlioB"):
            values = cur[f"delta_P_{target}"].to_numpy(float)
            frac_delta_values = cur[f"delta_frac_{target}"].to_numpy(float)
            frac_log2_values = cur[f"log2FC_{target}_fraction"].to_numpy(float)
            if len(frac_delta_values) > 1 and np.nanstd(frac_delta_values) > 1e-12:
                _, pvalue = stats.ttest_1samp(frac_delta_values, popmean=0.0, nan_policy="omit")
            else:
                pvalue = 1.0
            row[f"mean_delta_P_{target}"] = float(np.nanmean(values))
            row[f"mean_delta_frac_{target}"] = float(np.nanmean(frac_delta_values))
            row[f"mean_log2FC_{target}_fraction"] = float(np.nanmean(frac_log2_values))
            row[f"pvalue_{target}"] = float(pvalue) if np.isfinite(pvalue) else 1.0
        row["effect_abs_max"] = float(
            max(abs(row["mean_delta_P_NeuB"]), abs(row["mean_delta_P_GlioB"]))
        )
        row["effect_abs_max_fraction"] = float(
            max(abs(row["mean_delta_frac_NeuB"]), abs(row["mean_delta_frac_GlioB"]))
        )
        row["best_target"] = "NeuB" if abs(row["mean_delta_frac_NeuB"]) >= abs(row["mean_delta_frac_GlioB"]) else "GlioB"
        rows.append(row)
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    for target in ("NeuB", "GlioB"):
        ranked[f"fdr_{target}"] = fdr_bh(ranked[f"pvalue_{target}"].to_numpy(float))
    return ranked.sort_values("effect_abs_max_fraction", ascending=False).reset_index(drop=True)


def build_critical_sensitivity(summary_df: pd.DataFrame) -> pd.DataFrame:
    screen = summary_df[(summary_df["phase"] == "screen") & (summary_df["dose"] != 0)].copy()
    rows = []
    for source_time, cur in screen.groupby("source_time"):
        abs_shift = cur[["delta_P_NeuB", "delta_P_GlioB"]].abs().max(axis=1)
        abs_frac_shift = cur[["delta_frac_NeuB", "delta_frac_GlioB"]].abs().max(axis=1)
        rows.append(
            {
                "source_time": float(source_time),
                "mean_abs_fate_shift": float(abs_shift.mean()),
                "max_abs_fate_shift": float(abs_shift.max()),
                "mean_abs_delta_P_NeuB": float(cur["delta_P_NeuB"].abs().mean()),
                "mean_abs_delta_P_GlioB": float(cur["delta_P_GlioB"].abs().mean()),
                "max_abs_delta_P_NeuB": float(cur["delta_P_NeuB"].abs().max()),
                "max_abs_delta_P_GlioB": float(cur["delta_P_GlioB"].abs().max()),
                "mean_abs_fraction_shift": float(abs_frac_shift.mean()),
                "max_abs_fraction_shift": float(abs_frac_shift.max()),
                "mean_abs_delta_frac_NeuB": float(cur["delta_frac_NeuB"].abs().mean()),
                "mean_abs_delta_frac_GlioB": float(cur["delta_frac_GlioB"].abs().mean()),
                "max_abs_delta_frac_NeuB": float(cur["delta_frac_NeuB"].abs().max()),
                "max_abs_delta_frac_GlioB": float(cur["delta_frac_GlioB"].abs().max()),
            }
        )
    return pd.DataFrame(rows).sort_values("source_time")


def build_literature_annotation(genes: list[str], adata: ad.AnnData, raw_data: Path) -> pd.DataFrame:
    raw_present: set[str] = set()
    if raw_data.exists():
        try:
            raw = ad.read_h5ad(raw_data, backed="r")
            raw_present = set(map(str, raw.var_names))
            raw.file.close()
        except Exception:
            raw_present = set()
    roles = {
        "Mki67": ("RGC/proliferation", "MOSTA marker"),
        "Dbx1": ("RGC/progenitor", "MOSTA marker"),
        "Npas3": ("NeuB/neuroblast", "MOSTA marker"),
        "Adgrb3": ("NeuB/neuroblast", "MOSTA marker"),
        "Aldh1l1": ("GlioB/glioblast", "MOSTA/STORIES marker"),
        "Efna5": ("GlioB/glioblast", "MOSTA marker"),
        "Neurod1": ("NeuB/neurogenic", "neural differentiation marker"),
        "Sox2": ("RGC/progenitor", "literature marker"),
        "Pax6": ("RGC/progenitor", "literature marker"),
        "Ascl1": ("neurogenic progenitor", "literature marker"),
        "Neurog2": ("neurogenic progenitor", "literature marker"),
        "Gfap": ("glial", "literature marker"),
        "Olig2": ("glial/oligodendrocyte lineage", "literature marker"),
    }
    rows = []
    all_genes = sorted(set(genes) | set(DEFAULT_MARKERS))
    model_genes = set(map(str, adata.var_names))
    for gene in all_genes:
        role, source = roles.get(gene, ("screen candidate", "model screen"))
        rows.append(
            {
                "gene": gene,
                "role": role,
                "annotation_source": source,
                "in_current_2000_HVG_model": gene in model_genes,
                "in_raw_full_gene_data": gene in raw_present if raw_present else None,
                "can_be_directly_perturbed": gene in model_genes,
            }
        )
    return pd.DataFrame(rows)


def save_png_pdf(fig: plt.Figure, path_without_ext: Path) -> dict[str, str]:
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    png = path_without_ext.with_suffix(".png")
    pdf = path_without_ext.with_suffix(".pdf")
    fig.savefig(png, dpi=280, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def plot_critical_sensitivity(df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(
        df["source_time"],
        df["mean_abs_delta_frac_NeuB"],
        marker="o",
        ms=5,
        lw=2.0,
        color=COLORS["NeuB"],
        label="NeuB mean |delta fraction|",
    )
    ax.plot(
        df["source_time"],
        df["mean_abs_delta_frac_GlioB"],
        marker="o",
        ms=5,
        lw=2.0,
        color=COLORS["GlioB"],
        label="GlioB mean |delta fraction|",
    )
    ax.plot(df["source_time"], df["max_abs_fraction_shift"], marker="o", ms=5, lw=2.2, color="#222222", label="max fraction shift")
    ax.axvspan(13.5, 13.7, color="#d9d9d9", alpha=0.45, lw=0, label="critical window")
    ax.set_xlabel("Perturbation source time")
    ax.set_ylabel("Cell-type proportion response")
    ax.set_title("Critical-window cell-type proportion sensitivity")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, handlelength=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    return save_png_pdf(fig, out_dir / "critical_sensitivity_by_time")


def plot_volcano(ranked: pd.DataFrame, target: str, out_dir: Path) -> dict[str, str]:
    if ranked.empty:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.text(0.5, 0.5, "No screen results", ha="center", va="center")
        return save_png_pdf(fig, out_dir / f"gene_volcano_{target}")
    best = ranked.sort_values(f"mean_delta_frac_{target}", key=lambda s: s.abs(), ascending=False).drop_duplicates("gene")
    x = best[f"mean_log2FC_{target}_fraction"].to_numpy(float)
    fdr = best[f"fdr_{target}"].to_numpy(float)
    y = -np.log10(np.clip(fdr, 1e-12, 1.0))
    color = np.where(best[f"mean_delta_frac_{target}"] >= 0, COLORS[target], "#777777")
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(x, y, s=28, c=color, alpha=0.78, linewidths=0)
    ax.axvline(0.0, color="#999999", lw=1)
    ax.axhline(-np.log10(0.05), color="#999999", lw=1, ls="--")
    for _, row in best.head(8).iterrows():
        ax.text(
            float(row[f"mean_log2FC_{target}_fraction"]),
            -math.log10(max(float(row[f"fdr_{target}"]), 1e-12)),
            str(row["gene"]),
            fontsize=7,
        )
    ax.set_xlabel(f"log2FC {target} fraction")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Gene perturbation screen: {target} proportion")
    return save_png_pdf(fig, out_dir / f"gene_volcano_{target}")


def plot_top_gene_dose_response(
    summary_df: pd.DataFrame,
    top_genes: list[str],
    out_dir: Path,
    critical_time: float,
    args: argparse.Namespace,
) -> dict[str, str]:
    cur = summary_df[(summary_df["phase"] == "dose_response") & (summary_df["gene"].isin(top_genes))].copy()
    if cur.empty:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No dose-response results", ha="center", va="center")
        return save_png_pdf(fig, out_dir / "top_gene_dose_response")
    source = min(cur["source_time"].unique(), key=lambda v: abs(float(v) - float(critical_time)))
    cur = cur[np.isclose(cur["source_time"], source)]
    agg = cur.groupby(["gene", "dose"], as_index=False)[["delta_frac_NeuB", "delta_frac_GlioB"]].mean()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for gene, sub in agg.groupby("gene"):
        sub = sub.sort_values("dose")
        target = "GlioB" if sub["delta_frac_GlioB"].abs().max() >= sub["delta_frac_NeuB"].abs().max() else "NeuB"
        ax.plot(sub["dose"], sub[f"delta_frac_{target}"], marker="o", ms=7, lw=2.0, label=f"{gene}->{target}")
    ax.axhline(0.0, color="#999999", lw=1)
    if args.perturbation_mode == "latent_add":
        ax.set_xlabel(f"signed dose for z+x perturbation (|dose|=2 -> ||x||={float(args.latent_add_x):g})")
    else:
        ax.set_xlabel("z-score expression perturbation dose")
    ax.set_ylabel("Delta terminal cell-type fraction")
    ax.set_title(f"Top-gene dose response by cell-type proportion at E{source:g}")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return save_png_pdf(fig, out_dir / "top_gene_dose_response")


def plot_top_gene_fate_shift_bar(ranked: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    top = ranked.head(15).copy()
    if top.empty:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No screen results", ha="center", va="center")
        return save_png_pdf(fig, out_dir / "top_gene_fate_shift_bar")
    x = np.arange(top.shape[0])
    labels = [f"{row.gene} {row.dose:+g}\nE{row.source_time:g}" for row in top.itertuples()]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(x - 0.18, top["mean_delta_frac_NeuB"], width=0.36, color=COLORS["NeuB"], label="NeuB")
    ax.bar(x + 0.18, top["mean_delta_frac_GlioB"], width=0.36, color=COLORS["GlioB"], label="GlioB")
    ax.axhline(0.0, color="#999999", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean delta terminal cell-type fraction")
    ax.set_title("Top gene perturbation cell-type proportion shifts")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return save_png_pdf(fig, out_dir / "top_gene_fate_shift_bar")


def plot_top_gene_fate_heatmap(ranked: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    top = ranked.head(16).copy()
    if top.empty:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.text(0.5, 0.5, "No screen results", ha="center", va="center")
        return save_png_pdf(fig, out_dir / "top_gene_fate_heatmap")
    values = top[["mean_delta_frac_NeuB", "mean_delta_frac_GlioB"]].to_numpy(float)
    limit = max(float(np.nanmax(np.abs(values))), 1e-4)
    labels = [f"{row.gene} {row.dose:+g} E{row.source_time:g}" for row in top.itertuples()]
    fig, ax = plt.subplots(figsize=(5.2, max(3.6, 0.26 * len(labels) + 1.2)))
    im = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NeuB", "GlioB"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Top gene x fate cell-type proportion response")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Mean delta terminal cell-type fraction")
    return save_png_pdf(fig, out_dir / "top_gene_fate_heatmap")


def plot_local_effect_spatial(
    *,
    top_row: pd.Series | None,
    source_state: np.ndarray | None,
    local_effect: np.ndarray | None,
    out_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    if source_state is None or local_effect is None:
        ax.text(0.5, 0.5, "No local effect results", ha="center", va="center")
        return save_png_pdf(fig, out_dir / "local_perturbation_effect_spatial_map")
    vmax = float(np.nanquantile(local_effect, 0.98)) if np.isfinite(local_effect).any() else 1.0
    vmax = max(vmax, 1e-4)
    sc = ax.scatter(source_state[:, 0], source_state[:, 1], s=24, c=local_effect, cmap="magma", vmin=0.0, vmax=vmax, linewidths=0)
    title = "Local gene-potential velocity effect"
    if top_row is not None:
        title = f"{top_row['gene']} dose {top_row['dose']:g} at E{top_row['source_time']:g}"
    ax.set_title(title)
    ax.set_xlabel("spatial 1")
    ax.set_ylabel("spatial 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("1 - cos(v_z control, v_z perturbed)")
    return save_png_pdf(fig, out_dir / "local_perturbation_effect_spatial_map")


def plot_baseline_keyframes(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    controls: dict[tuple[float, int], ControlPack],
    source_pairs: list[tuple[float, float]],
    terminal_input: float,
    spatial_dim: int,
    classifier: torch.nn.Module,
    label_map: dict[int, str],
    out_dir: Path,
) -> dict[str, str]:
    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    raw_time = np.asarray(adata.obs[args.raw_time_key], dtype=np.float64)
    time_input = np.asarray(adata.obs[args.time_input_key], dtype=np.float64)
    labels = adata.obs[args.annotation_key].astype(str).to_numpy()
    spatial = np.asarray(adata.obsm[args.spatial_key], dtype=np.float32)
    for t in sorted(np.unique(raw_time)):
        if np.isclose(t, [12.5, 14.5, 16.5]).any():
            mask = np.isclose(raw_time, t)
            panels.append((f"E{t:g} real", spatial[mask], labels[mask]))
    for source_time, input_time in source_pairs:
        key = (float(source_time), 0)
        if key not in controls:
            continue
        state = controls[key].source_state
        _, hard, _ = classifier_readout(
            classifier,
            label_map,
            state,
            float(input_time),
            spatial_dim,
            batch_size=int(args.classifier_batch_size),
        )
        panels.append((f"E{source_time:g} generated", state[:, :spatial_dim].detach().cpu().numpy(), hard))
    if controls:
        first_control = next(iter(controls.values()))
        state = first_control.control_rollout["state"]
        _, hard, _ = classifier_readout(
            classifier,
            label_map,
            state,
            float(terminal_input),
            spatial_dim,
            batch_size=int(args.classifier_batch_size),
        )
        panels.append(("E16.5 generated", state[:, :spatial_dim].detach().cpu().numpy(), hard))
    order = {"E12.5 real": 0, "E13 generated": 1, "E13.6 generated": 2, "E14 generated": 3, "E14.5 real": 4, "E16.5 generated": 5, "E16.5 real": 6}
    panels = sorted(panels, key=lambda item: order.get(item[0], 99))
    if not panels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No keyframes available", ha="center", va="center")
        return save_png_pdf(fig, out_dir / "baseline_real_generated_keyframes")
    ncols = min(4, len(panels))
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)
    for ax, (title, pts, panel_labels) in zip(axes.ravel(), panels):
        for label in CELL_TYPES:
            mask = np.asarray(panel_labels, dtype=object) == label
            if mask.any():
                ax.scatter(pts[mask, 0], pts[mask, 1], s=7, c=COLORS[label], alpha=0.78, linewidths=0)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[label], markersize=6, label=label)
        for label in CELL_TYPES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(CELL_TYPES), frameon=False)
    fig.suptitle("Dorsal midbrain baseline real/generated keyframes", y=1.02)
    return save_png_pdf(fig, out_dir / "baseline_real_generated_keyframes")


def plot_terminal_spatial(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    top_row: pd.Series | None,
    top_state: np.ndarray | None,
    base_state: np.ndarray | None,
    top_labels: np.ndarray | None,
    terminal_input: float,
    out_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    time_input = np.asarray(adata.obs[args.time_input_key], dtype=np.float64)
    terminal_mask = np.isclose(time_input, terminal_input, atol=1e-4)
    observed = np.asarray(adata.obsm[args.spatial_key][terminal_mask], dtype=np.float32)
    ax.scatter(observed[:, 0], observed[:, 1], s=8, c="#d0d0d0", alpha=0.28, linewidths=0, label="observed E16.5")
    if base_state is not None:
        ax.scatter(base_state[:, 0], base_state[:, 1], s=12, c="#333333", alpha=0.42, linewidths=0, label="control")
    if top_state is not None and top_labels is not None:
        for label in CELL_TYPES:
            mask = top_labels == label
            ax.scatter(top_state[mask, 0], top_state[mask, 1], s=20, c=COLORS[label], alpha=0.86, linewidths=0, label=f"perturbed {label}")
    title = "Top terminal spatial perturbation"
    if top_row is not None:
        title = f"{top_row['gene']} dose {top_row['dose']:g} at E{top_row['source_time']:g}"
    ax.set_title(title)
    ax.set_xlabel("spatial 1")
    ax.set_ylabel("spatial 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    return save_png_pdf(fig, out_dir / "terminal_spatial_top_perturbation")


def _slice_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)


def _draw_observed_slice_background(
    ax: plt.Axes,
    observed: np.ndarray,
    observed_labels: np.ndarray,
    *,
    colored: bool,
    point_size: float = 8.0,
) -> None:
    if colored:
        for label in CELL_TYPES:
            mask = observed_labels == label
            if mask.any():
                ax.scatter(
                    observed[mask, 0],
                    observed[mask, 1],
                    s=point_size,
                    c=COLORS[label],
                    alpha=0.18,
                    linewidths=0,
                    label=f"real {label}",
                )
    else:
        ax.scatter(observed[:, 0], observed[:, 1], s=point_size, c="#cfcfcf", alpha=0.25, linewidths=0, label="real E16.5")


def plot_slice_terminal_counterfactual(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    top_row: pd.Series | None,
    top_state: np.ndarray | None,
    base_state: np.ndarray | None,
    top_labels: np.ndarray | None,
    base_labels: np.ndarray | None,
    terminal_input: float,
    out_dir: Path,
    output_name: str = "slice_terminal_counterfactual",
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    observed, observed_labels = observed_slice_coords(adata, args, terminal_input)
    _draw_observed_slice_background(ax, observed, observed_labels, colored=True, point_size=8.0)
    if base_state is not None:
        base_xy = spatial_to_slice_coords(adata, args, base_state[:, :2])
        ax.scatter(base_xy[:, 0], base_xy[:, 1], s=15, c="#2b2b2b", alpha=0.35, linewidths=0, label="control")
    if top_state is not None and top_labels is not None:
        top_xy = spatial_to_slice_coords(adata, args, top_state[:, :2])
        changed = np.zeros(top_xy.shape[0], dtype=bool)
        if base_labels is not None:
            changed = np.asarray(base_labels, dtype=object) != np.asarray(top_labels, dtype=object)
        for label in CELL_TYPES:
            mask = np.asarray(top_labels, dtype=object) == label
            if mask.any():
                ax.scatter(
                    top_xy[mask, 0],
                    top_xy[mask, 1],
                    s=np.where(changed[mask], 42, 24),
                    c=COLORS[label],
                    alpha=0.88,
                    linewidths=np.where(changed[mask], 0.8, 0.0),
                    edgecolors=np.where(changed[mask], "#111111", COLORS[label]),
                    label=f"perturbed {label}",
                )
    title = "Terminal perturbation on E16.5 ST slice"
    if top_row is not None:
        title = f"{top_row['gene']} dose {top_row['dose']:g} at E{top_row['source_time']:g}: terminal ST-slice readout"
    _slice_axes(ax, title)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    return save_png_pdf(fig, out_dir / output_name)


def plot_slice_fate_switch_map(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    top_row: pd.Series | None,
    top_state: np.ndarray | None,
    base_state: np.ndarray | None,
    top_labels: np.ndarray | None,
    base_labels: np.ndarray | None,
    terminal_input: float,
    out_dir: Path,
    output_name: str = "slice_fate_switch_map",
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    observed, observed_labels = observed_slice_coords(adata, args, terminal_input)
    _draw_observed_slice_background(ax, observed, observed_labels, colored=False, point_size=8.0)
    if top_state is not None and top_labels is not None and base_labels is not None:
        top_xy = spatial_to_slice_coords(adata, args, top_state[:, :2])
        base_xy = spatial_to_slice_coords(adata, args, base_state[:, :2]) if base_state is not None else top_xy
        changed = np.asarray(base_labels, dtype=object) != np.asarray(top_labels, dtype=object)
        if changed.any():
            displacement = top_xy - base_xy
            step = max(1, int(math.ceil(int(changed.sum()) / 80)))
            changed_idx = np.flatnonzero(changed)[::step]
            ax.quiver(
                base_xy[changed_idx, 0],
                base_xy[changed_idx, 1],
                displacement[changed_idx, 0],
                displacement[changed_idx, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0028,
                color="#222222",
                alpha=0.55,
            )
            for label in CELL_TYPES:
                mask = changed & (np.asarray(top_labels, dtype=object) == label)
                if mask.any():
                    ax.scatter(
                        top_xy[mask, 0],
                        top_xy[mask, 1],
                        s=46,
                        c=COLORS[label],
                        alpha=0.92,
                        edgecolors="#111111",
                        linewidths=0.7,
                        label=f"switch to {label}",
                    )
        else:
            ax.text(0.5, 0.5, "No hard fate switches", transform=ax.transAxes, ha="center", va="center")
    title = "Spatial map of hard fate switches"
    if top_row is not None:
        title = f"{top_row['gene']} dose {top_row['dose']:g} at E{top_row['source_time']:g}: hard fate switches"
    _slice_axes(ax, title)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    return save_png_pdf(fig, out_dir / output_name)


def plot_slice_delta_probability(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    top_row: pd.Series | None,
    top_state: np.ndarray | None,
    delta_prob: np.ndarray | None,
    label: str,
    terminal_input: float,
    out_dir: Path,
    output_name: str | None = None,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    observed, observed_labels = observed_slice_coords(adata, args, terminal_input)
    _draw_observed_slice_background(ax, observed, observed_labels, colored=False, point_size=8.0)
    if top_state is not None and delta_prob is not None:
        xy = spatial_to_slice_coords(adata, args, top_state[:, :2])
        values = np.asarray(delta_prob, dtype=np.float32)
        vmax = max(float(np.nanquantile(np.abs(values), 0.98)), 1e-3)
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=30,
            c=values,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.92,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
        cbar.set_label(f"delta P({label}) perturbed - control")
    title = f"Spatial delta P({label}) on E16.5 ST slice"
    if top_row is not None:
        title = f"{top_row['gene']} dose {top_row['dose']:g} at E{top_row['source_time']:g}: delta P({label})"
    _slice_axes(ax, title)
    return save_png_pdf(fig, out_dir / (output_name or f"slice_delta_P_{label}"))


def plot_potential_response(summary_df: pd.DataFrame, top_genes: list[str], out_dir: Path) -> dict[str, str]:
    cur = summary_df[(summary_df["gene"].isin(top_genes)) & (summary_df["dose"] != 0)].copy()
    if cur.empty:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No perturbation results", ha="center", va="center")
        return save_png_pdf(fig, out_dir / "potential_response_top_genes")
    agg = cur.groupby("gene", as_index=False)[["delta_terminal_U_mean", "integrated_abs_step_dU_median"]].mean()
    agg = agg.set_index("gene").loc[[gene for gene in top_genes if gene in set(agg["gene"])]].reset_index()
    x = np.arange(agg.shape[0])
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.18, agg["delta_terminal_U_mean"], width=0.36, color="#8a3ffc", label="delta terminal U")
    ax2.bar(x + 0.18, agg["integrated_abs_step_dU_median"], width=0.36, color="#222222", alpha=0.65, label="integrated |step dU|")
    ax1.axhline(0.0, color="#999999", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg["gene"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean delta terminal U")
    ax2.set_ylabel("Mean integrated |step dU|")
    ax1.set_title("Potential response for top perturbation genes")
    return save_png_pdf(fig, out_dir / "potential_response_top_genes")


def plot_marker_availability(annotation_df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    markers = annotation_df[annotation_df["gene"].isin(DEFAULT_MARKERS)].copy()
    markers = markers.sort_values(["can_be_directly_perturbed", "gene"], ascending=[False, True])
    y = np.arange(markers.shape[0])
    colors = np.where(markers["can_be_directly_perturbed"], "#2ca25f", "#bdbdbd")
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.barh(y, markers["can_be_directly_perturbed"].astype(int), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(markers["gene"], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["missing", "in model"])
    ax.set_title("Literature marker availability in current 2000-HVG model")
    return save_png_pdf(fig, out_dir / "literature_marker_availability")


def plot_overview_montage(figures: dict[str, dict[str, str]], out_dir: Path) -> dict[str, str]:
    preferred = [
        "baseline_real_generated_keyframes",
        "critical_sensitivity_by_time",
        "top_gene_fate_heatmap",
        "slice_terminal_counterfactual",
        "slice_fate_switch_map",
        "slice_delta_P_GlioB",
        "local_perturbation_effect_spatial_map",
        "potential_response_top_genes",
    ]
    images: list[tuple[str, np.ndarray]] = []
    for key in preferred:
        png = figures.get(key, {}).get("png")
        if png and Path(png).exists():
            images.append((key, plt.imread(png)))
    if not images:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No overview figures available", ha="center", va="center")
        return save_png_pdf(fig, out_dir / "overview_montage")
    ncols = 2
    nrows = int(math.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.6 * nrows), squeeze=False)
    for ax, (key, image) in zip(axes.ravel(), images):
        ax.imshow(image)
        ax.set_title(key.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(images) :]:
        ax.axis("off")
    fig.suptitle("RegVelo-style dorsal perturbation overview", y=0.995)
    return save_png_pdf(fig, out_dir / "overview_montage")


def write_interpretation(
    *,
    out_dir: Path,
    sensitivity: pd.DataFrame,
    ranked: pd.DataFrame,
    local_effect: pd.DataFrame,
    fate_enrichment: pd.DataFrame,
    marker_df: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    lines: list[str] = []
    lines.append("# Dorsal midbrain RegVelo-style perturbation criticality interpretation\n")
    lines.append("本分析是 model-inferred in silico perturbation，不代表真实 CRISPR 或实验扰动。\n")
    mode = summary.get("perturbation_mode", "unknown")
    if mode == "latent_add":
        lines.append(
            "- 扰动采用温和的 z+x latent shift：gene direction 由 GAT encoder 从小幅表达改变估计，"
            f"|dose|=2 时 ||x||={summary.get('latent_add_x', 'NA')}，并封顶到 "
            f"{summary.get('latent_add_max_x', 'NA')}，避免过大扰动。\n"
        )
    if not sensitivity.empty:
        best = sensitivity.sort_values("max_abs_fraction_shift", ascending=False).iloc[0]
        lines.append(
            f"- 扰动敏感性最高的 source time 是 E{best['source_time']:.2f}，"
            f"max_abs_fraction_shift={best['max_abs_fraction_shift']:.4f}。\n"
        )
        critical = sensitivity[(sensitivity["source_time"] >= 13.5) & (sensitivity["source_time"] <= 13.7)]
        pre = sensitivity[sensitivity["source_time"] < 13.5]
        post = sensitivity[sensitivity["source_time"] >= 14.0]
        if not critical.empty:
            critical_peak = float(critical["max_abs_fraction_shift"].max())
            pre_peak = float(pre["max_abs_fraction_shift"].max()) if not pre.empty else float("-inf")
            post_peak = float(post["max_abs_fraction_shift"].max()) if not post.empty else float("-inf")
            if critical_peak >= max(pre_peak, post_peak) * 0.95:
                lines.append("- 当前结果支持临界窗口附近是扰动敏感区，且响应接近或达到所有 source time 的最高值。\n")
            elif critical_peak > post_peak:
                lines.append("- 当前结果支持临界窗口附近比 E14.0 后更敏感，但最高响应出现在更早时间；不能写成严格的临界峰值。\n")
            else:
                lines.append("- 当前结果没有强支持临界窗口显著高于临界前/后；需要把结论写成模型推断不稳定或效应较弱。\n")
    if not local_effect.empty:
        top_local = local_effect.sort_values("local_effect_gene_mean", ascending=False).iloc[0]
        lines.append(
            f"- RegVelo-style local effect 最高的是 {top_local['gene']} dose {top_local['dose']:+.1f} "
            f"at E{top_local['source_time']:.2f}，mean 1-cos(v_z)={top_local['local_effect_gene_mean']:.4f}。\n"
        )
    if not fate_enrichment.empty:
        cls = fate_enrichment[fate_enrichment["readout"] == "classifier"].copy()
        cls = cls[cls["terminal_state"].isin(["NeuB", "GlioB"])]
        if not cls.empty:
            top_fate = cls.sort_values("delta_mean_probability", key=lambda s: s.abs(), ascending=False).iloc[0]
            lines.append(
                f"- RegVelo-style fate enrichment 最强的是 {top_fate['gene']} dose {top_fate['dose']:+.1f} "
                f"toward {top_fate['terminal_state']}，delta mean probability={top_fate['delta_mean_probability']:.4f}。\n"
            )
    if not ranked.empty:
        glio = ranked.sort_values("mean_delta_frac_GlioB", ascending=False).head(8)
        neu = ranked.sort_values("mean_delta_frac_NeuB", ascending=False).head(8)
        lines.append("\n## Genes pushing GlioB in the model\n")
        for _, row in glio.iterrows():
            lines.append(
                f"- {row['gene']}: E{row['source_time']:.2f}, dose {row['dose']:+.1f}, "
                f"delta_frac_GlioB={row['mean_delta_frac_GlioB']:.4f}, FDR={row['fdr_GlioB']:.3g}\n"
            )
        lines.append("\n## Genes pushing NeuB in the model\n")
        for _, row in neu.iterrows():
            lines.append(
                f"- {row['gene']}: E{row['source_time']:.2f}, dose {row['dose']:+.1f}, "
                f"delta_frac_NeuB={row['mean_delta_frac_NeuB']:.4f}, FDR={row['fdr_NeuB']:.3g}\n"
            )
    missing = marker_df[(marker_df["gene"].isin(DEFAULT_MARKERS)) & (~marker_df["can_be_directly_perturbed"])]
    if not missing.empty:
        lines.append("\n## Literature marker caveat\n")
        lines.append(
            "当前模型只使用 2000 HVG，因此一些 MOSTA/STORIES 重要 marker 不能被直接 perturb："
            + ", ".join(map(str, missing["gene"].tolist()))
            + "。这些基因需要 marker-preserved GAT/hybrid rerun 才能做直接扰动结论。\n"
        )
    lines.append("\n## Method\n")
    if mode == "latent_add":
        lines.append(
            "每个 source-time/seed 先从 E12.5 RGC-origin rollout 到 source time，"
            "将 generated GAT latent decode 到表达，用该基因的小幅表达改变估计 GAT latent direction，"
            "再按 z+x 的形式加入小范数 latent shift，最后用 hybrid dynamics rollout 到 E16.5。"
            "主要 fate readout 来自 stVCR spatiotemporal classifier，KNN31 state+time label transfer 作为验证。\n"
        )
    else:
        lines.append(
            "每个 source-time/seed 先从 E12.5 RGC-origin rollout 到 source time，"
            "将 generated GAT latent decode 到表达，做 gene z-score dose，再 encode 回 GAT latent，"
            "最后用 hybrid dynamics rollout 到 E16.5。主要 fate readout 来自 stVCR spatiotemporal classifier，"
            "KNN31 state+time label transfer 作为验证。\n"
        )
    lines.append("\n## Run metadata\n")
    lines.append("```json\n")
    lines.append(json.dumps(summary, indent=2, ensure_ascii=False))
    lines.append("\n```\n")
    path = out_dir / "REGVELO_STYLE_PERTURBATION_INTERPRETATION.md"
    text = "".join(lines)
    path.write_text(text, encoding="utf-8")
    (out_dir / "PERTURBATION_INTERPRETATION.md").write_text(text, encoding="utf-8")
    return path


def choose_genes(
    args: argparse.Namespace,
    prescreen: pd.DataFrame,
    adata: ad.AnnData,
) -> list[str]:
    present = set(map(str, adata.var_names))
    if args.genes:
        return [gene for gene in args.genes if gene in present]
    if args.smoke:
        return [gene for gene in FORCED_GENES if gene in present]
    top = prescreen.head(int(args.top_screen_genes))["gene"].astype(str).tolist()
    for gene in FORCED_GENES:
        if gene in present and gene not in top:
            top.append(gene)
    return top


def run_condition(
    *,
    phase: str,
    gene: str,
    gene_idx: int,
    dose: float,
    seed: int,
    source_time: float,
    source_input: float,
    terminal_time: float,
    terminal_input: float,
    control: ControlPack,
    model: stVCR_DynamicModel,
    gat: GATBundle,
    classifier: torch.nn.Module,
    label_map: dict[int, str],
    knn: KNNReadout,
    expr_stats: dict[str, np.ndarray],
    spatial_dim: int,
    method: str,
    step_size: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    condition = {
        "dataset": "dorsal_midbrain",
        "phase": phase,
        "source_time": float(source_time),
        "source_time_input": float(source_input),
        "terminal_time": float(terminal_time),
        "terminal_time_input": float(terminal_input),
        "gene": str(gene),
        "dose": float(dose),
        "seed": int(seed),
    }
    if abs(float(dose)) < 1e-12:
        metrics = control.control_metrics
        row = row_for_condition(
            phase=phase,
            gene=gene,
            dose=float(dose),
            seed=seed,
            source_time=source_time,
            source_input=source_input,
            terminal_time=terminal_time,
            terminal_input=terminal_input,
            metrics=metrics,
            baseline_metrics=control.control_metrics,
        )
        row.update(
            {
                "perturbation_mode": str(args.perturbation_mode),
                "latent_add_x": 0.0,
                "latent_direction_expression_dose": 0.0,
                "latent_direction_delta_norm_mean": 0.0,
                "latent_direction_delta_norm_median": 0.0,
                "latent_shift_norm_mean": 0.0,
                "latent_shift_norm_median": 0.0,
            }
        )
        for key in (
            "local_effect_spatial_mean",
            "local_effect_spatial_median",
            "local_effect_spatial_q90",
            "local_effect_gene_mean",
            "local_effect_gene_median",
            "local_effect_gene_q90",
            "local_effect_joint_mean",
            "local_effect_joint_median",
            "local_effect_joint_q90",
            "local_delta_velocity_spatial_norm_mean",
            "local_delta_velocity_gene_norm_mean",
        ):
            row[key] = 0.0
        return row, None, None, None, None, None, None, None, None, [], []
    perturb_info: dict[str, float]
    if args.perturbation_mode == "expression_encode":
        pert_expr = perturb_expression(control.decoded_expression, gene_idx, dose, expr_stats, device)
        pert_gene = encode_expression(gat, pert_expr, batch_size=int(args.gat_batch_size), device=device)
        latent_delta = pert_gene - control.control_state[:, spatial_dim:]
        latent_delta_norm = torch.linalg.norm(latent_delta, dim=1)
        perturb_info = {
            "latent_add_x": float("nan"),
            "latent_direction_expression_dose": float(dose),
            "latent_direction_delta_norm_mean": float(latent_delta_norm.detach().cpu().mean().item()),
            "latent_direction_delta_norm_median": float(latent_delta_norm.detach().cpu().median().item()),
            "latent_shift_norm_mean": float(latent_delta_norm.detach().cpu().mean().item()),
            "latent_shift_norm_median": float(latent_delta_norm.detach().cpu().median().item()),
        }
    else:
        pert_gene, perturb_info = gene_latent_add_shift(
            control=control,
            gene_idx=gene_idx,
            dose=float(dose),
            gat=gat,
            expr_stats=expr_stats,
            spatial_dim=spatial_dim,
            args=args,
            device=device,
        )
    pert_state = control.control_state.detach().clone()
    pert_state[:, spatial_dim:] = pert_gene
    base_velocity, _ = dynamics_vector(
        model,
        control.control_state,
        control.source_logw,
        source_input,
        chunk_size=int(args.ode_chunk_size),
        device=device,
    )
    pert_velocity, _ = dynamics_vector(
        model,
        pert_state,
        control.source_logw,
        source_input,
        chunk_size=int(args.ode_chunk_size),
        device=device,
    )
    local_stats = vector_effect_stats(base_velocity, pert_velocity, spatial_dim)
    rollout = rollout_path(
        model,
        pert_state,
        control.source_logw,
        source_input,
        terminal_input,
        method=method,
        step_size=step_size,
        metric_dt=float(args.metric_dt),
        chunk_size=int(args.ode_chunk_size),
        device=device,
    )
    metrics = summarize_rollout(
        rollout=rollout,
        reference_rollout=control.control_rollout,
        classifier=classifier,
        label_map=label_map,
        knn=knn,
        terminal_input=terminal_input,
        spatial_dim=spatial_dim,
        args=args,
        seed=int(args.seed) + int(seed),
    )
    row = row_for_condition(
        phase=phase,
        gene=gene,
        dose=float(dose),
        seed=seed,
        source_time=source_time,
        source_input=source_input,
        terminal_time=terminal_time,
        terminal_input=terminal_input,
        metrics=metrics,
        baseline_metrics=control.control_metrics,
    )
    row["perturbation_mode"] = str(args.perturbation_mode)
    row.update(perturb_info)
    row.update({key: value for key, value in local_stats.items() if not isinstance(value, np.ndarray)})
    local_row = {
        **condition,
        "perturbation_mode": str(args.perturbation_mode),
        "n_cells": int(control.control_state.shape[0]),
        **{key: value for key, value in local_stats.items() if not isinstance(value, np.ndarray)},
    }
    enrichment = fate_enrichment_rows(
        condition={**condition, "perturbation_mode": str(args.perturbation_mode)},
        metrics=metrics,
        baseline_metrics=control.control_metrics,
    )
    classes = list(metrics.get("_classifier_classes", []))
    pert_probs = np.asarray(metrics.get("_classifier_probs"), dtype=np.float32)
    base_probs = np.asarray(control.control_metrics.get("_classifier_probs"), dtype=np.float32)

    def delta_probability(label: str) -> np.ndarray:
        if label not in classes or pert_probs.ndim != 2 or base_probs.ndim != 2:
            return np.full((pert_state.shape[0],), np.nan, dtype=np.float32)
        idx = classes.index(label)
        return (pert_probs[:, idx] - base_probs[:, idx]).astype(np.float32)

    return (
        row,
        rollout["state"].detach().cpu().numpy().astype(np.float32),
        control.control_rollout["state"].detach().cpu().numpy().astype(np.float32),
        np.asarray(metrics["_classifier_hard"], dtype=object),
        np.asarray(control.control_metrics["_classifier_hard"], dtype=object),
        delta_probability("NeuB"),
        delta_probability("GlioB"),
        control.control_state.detach().cpu().numpy().astype(np.float32),
        np.asarray(local_stats["local_effect_gene"], dtype=np.float32),
        [local_row],
        enrichment,
    )


def main() -> None:
    args = parse_args()
    seed_all(int(args.seed))
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))

    summary = load_json(run_dir / "summary.json")
    adata = ad.read_h5ad(summary["trained_h5ad"])
    adata.obs[args.annotation_key] = adata.obs[args.annotation_key].astype("category")
    if args.spatial_key not in adata.obsm:
        raise KeyError(f"Missing spatial key {args.spatial_key} in trained h5ad.")
    expression = dense_float32(adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X)
    expr_stats = expression_stats(expression, float(args.expression_clip_quantile))

    pairs = observed_time_pairs(adata, args)
    source_inputs = [raw_to_input(t, pairs) for t in args.source_times]
    terminal_input = raw_to_input(float(args.terminal_time), pairs)
    initial_input = float(np.min(np.asarray(adata.obs[args.time_input_key], dtype=np.float64)))
    spatial_dim = int(adata.obsm[args.spatial_key].shape[1])
    method = str(summary.get("config", {}).get("ode_method", "rk4"))
    step_size = float(summary.get("config", {}).get("ode_step_size", 0.25))

    model = load_dynamics(summary, adata, args, device)
    classifier, label_map, classifier_path = load_classifier(run_dir, device)
    gat = load_gat_bundle(adata, device, args.decoder_checkpoint)
    if list(map(str, adata.var_names)) != gat.gene_names:
        raise ValueError("Trained h5ad var_names do not match the GAT checkpoint gene order.")
    knn = build_terminal_knn(adata, args, terminal_input, spatial_dim, int(args.knn_k))

    controls: dict[tuple[float, int], ControlPack] = {}
    critical_source = min(zip(args.source_times, source_inputs), key=lambda pair: abs(float(pair[0]) - float(args.critical_time)))
    critical_indices = select_source_indices(adata, args, initial_input, int(args.seed))
    critical_control = make_control_pack(
        adata=adata,
        args=args,
        model=model,
        gat=gat,
        classifier=classifier,
        label_map=label_map,
        knn=knn,
        source_input=float(critical_source[1]),
        terminal_input=float(terminal_input),
        source_indices=critical_indices,
        initial_input=initial_input,
        method=method,
        step_size=step_size,
        spatial_dim=spatial_dim,
        expr_stats=expr_stats,
        device=device,
    )
    prescreen = prescreen_genes(
        control=critical_control,
        gat=gat,
        classifier=classifier,
        label_map=label_map,
        expr_stats=expr_stats,
        source_input=float(critical_source[1]),
        spatial_dim=spatial_dim,
        args=args,
        device=device,
    )
    prescreen_csv = out_dir / "gene_prescreen_scores.csv"
    prescreen.to_csv(prescreen_csv, index=False)

    screen_genes = choose_genes(args, prescreen, adata)
    if not screen_genes:
        raise ValueError("No screen genes selected.")
    gene_to_idx = {str(gene): int(idx) for idx, gene in enumerate(map(str, adata.var_names))}
    rows: list[dict[str, Any]] = []
    local_effect_rows: list[dict[str, Any]] = []
    fate_enrichment: list[dict[str, Any]] = []
    top_state: np.ndarray | None = None
    top_base_state: np.ndarray | None = None
    top_labels: np.ndarray | None = None
    top_base_labels: np.ndarray | None = None
    top_delta_prob_neub: np.ndarray | None = None
    top_delta_prob_gliob: np.ndarray | None = None
    top_source_state: np.ndarray | None = None
    top_local_effect: np.ndarray | None = None
    top_abs_effect = -math.inf
    top_row_for_plot: pd.Series | None = None
    top_by_source_time: dict[float, dict[str, Any]] = {}

    def condition_payload(
        *,
        row: dict[str, Any],
        state_np: np.ndarray | None,
        base_np: np.ndarray | None,
        labels_np: np.ndarray | None,
        base_labels_np: np.ndarray | None,
        delta_prob_neub_np: np.ndarray | None,
        delta_prob_gliob_np: np.ndarray | None,
        source_np: np.ndarray | None,
        local_np: np.ndarray | None,
    ) -> dict[str, Any]:
        return {
            "row": pd.Series(row),
            "state": state_np,
            "base_state": base_np,
            "labels": labels_np,
            "base_labels": base_labels_np,
            "delta_prob_neub": delta_prob_neub_np,
            "delta_prob_gliob": delta_prob_gliob_np,
            "source_state": source_np,
            "local_effect": local_np,
            "effect": max(abs(float(row["delta_frac_NeuB"])), abs(float(row["delta_frac_GlioB"]))),
        }

    def get_control(raw_time: float, input_time: float, seed_idx: int) -> ControlPack:
        key = (float(raw_time), int(seed_idx))
        if key not in controls:
            source_indices = select_source_indices(adata, args, initial_input, int(args.seed) + int(seed_idx))
            controls[key] = make_control_pack(
                adata=adata,
                args=args,
                model=model,
                gat=gat,
                classifier=classifier,
                label_map=label_map,
                knn=knn,
                source_input=float(input_time),
                terminal_input=float(terminal_input),
                source_indices=source_indices,
                initial_input=initial_input,
                method=method,
                step_size=step_size,
                spatial_dim=spatial_dim,
                expr_stats=expr_stats,
                device=device,
            )
        return controls[key]

    seeds = list(range(1 if args.smoke else int(args.seeds)))
    screen_doses = list(map(float, args.screen_doses))
    source_pairs = list(zip(args.source_times, source_inputs))
    if args.smoke:
        source_pairs = [critical_source]
        screen_doses = [-2.0, 2.0]

    for source_time, source_input in source_pairs:
        for seed_idx in seeds:
            control = get_control(float(source_time), float(source_input), int(seed_idx))
            for gene in screen_genes:
                gene_idx = gene_to_idx[gene]
                for dose in screen_doses:
                    (
                        row,
                        state_np,
                        base_np,
                        labels_np,
                        base_labels_np,
                        delta_prob_neub_np,
                        delta_prob_gliob_np,
                        source_np,
                        local_np,
                        local_rows,
                        enrichment_rows,
                    ) = run_condition(
                        phase="screen",
                        gene=gene,
                        gene_idx=gene_idx,
                        dose=float(dose),
                        seed=int(seed_idx),
                        source_time=float(source_time),
                        source_input=float(source_input),
                        terminal_time=float(args.terminal_time),
                        terminal_input=float(terminal_input),
                        control=control,
                        model=model,
                        gat=gat,
                        classifier=classifier,
                        label_map=label_map,
                        knn=knn,
                        expr_stats=expr_stats,
                        spatial_dim=spatial_dim,
                        method=method,
                        step_size=step_size,
                        args=args,
                        device=device,
                    )
                    rows.append(row)
                    local_effect_rows.extend(local_rows)
                    fate_enrichment.extend(enrichment_rows)
                    effect = max(abs(float(row["delta_frac_NeuB"])), abs(float(row["delta_frac_GlioB"])))
                    payload = condition_payload(
                        row=row,
                        state_np=state_np,
                        base_np=base_np,
                        labels_np=labels_np,
                        base_labels_np=base_labels_np,
                        delta_prob_neub_np=delta_prob_neub_np,
                        delta_prob_gliob_np=delta_prob_gliob_np,
                        source_np=source_np,
                        local_np=local_np,
                    )
                    by_time_key = float(source_time)
                    if state_np is not None and effect > float(top_by_source_time.get(by_time_key, {}).get("effect", -math.inf)):
                        top_by_source_time[by_time_key] = payload
                    if effect > top_abs_effect and state_np is not None:
                        top_abs_effect = effect
                        top_state = state_np
                        top_base_state = base_np
                        top_labels = labels_np
                        top_base_labels = base_labels_np
                        top_delta_prob_neub = delta_prob_neub_np
                        top_delta_prob_gliob = delta_prob_gliob_np
                        top_source_state = source_np
                        top_local_effect = local_np
                        top_row_for_plot = pd.Series(row)

    summary_df = pd.DataFrame(rows)
    ranked = build_ranked_screen(summary_df)
    ranked_csv = out_dir / "gene_screen_ranked.csv"
    ranked.to_csv(ranked_csv, index=False)

    if args.smoke:
        dose_genes = screen_genes
    elif args.genes:
        dose_genes = screen_genes[: int(args.top_dose_genes)]
    else:
        dose_genes = []
        for gene in ranked["gene"].astype(str).tolist():
            if gene not in dose_genes:
                dose_genes.append(gene)
            if len(dose_genes) >= int(args.top_dose_genes):
                break
    dose_response_source = critical_source
    dose_rows: list[dict[str, Any]] = []
    existing = {
        (row["phase"], row["gene"], float(row["source_time"]), float(row["dose"]), int(row["seed"]))
        for row in rows
    }
    for seed_idx in seeds:
        control = get_control(float(dose_response_source[0]), float(dose_response_source[1]), int(seed_idx))
        for gene in dose_genes:
            gene_idx = gene_to_idx[gene]
            for dose in map(float, args.dose_response_doses):
                key = ("dose_response", gene, float(dose_response_source[0]), float(dose), int(seed_idx))
                if key in existing:
                    continue
                (
                    row,
                    state_np,
                    base_np,
                    labels_np,
                    base_labels_np,
                    delta_prob_neub_np,
                    delta_prob_gliob_np,
                    source_np,
                    local_np,
                    local_rows,
                    enrichment_rows,
                ) = run_condition(
                    phase="dose_response",
                    gene=gene,
                    gene_idx=gene_idx,
                    dose=float(dose),
                    seed=int(seed_idx),
                    source_time=float(dose_response_source[0]),
                    source_input=float(dose_response_source[1]),
                    terminal_time=float(args.terminal_time),
                    terminal_input=float(terminal_input),
                    control=control,
                    model=model,
                    gat=gat,
                    classifier=classifier,
                    label_map=label_map,
                    knn=knn,
                    expr_stats=expr_stats,
                    spatial_dim=spatial_dim,
                    method=method,
                    step_size=step_size,
                    args=args,
                    device=device,
                )
                dose_rows.append(row)
                local_effect_rows.extend(local_rows)
                fate_enrichment.extend(enrichment_rows)
                effect = max(abs(float(row["delta_frac_NeuB"])), abs(float(row["delta_frac_GlioB"])))
                if effect > top_abs_effect and state_np is not None:
                    top_abs_effect = effect
                    top_state = state_np
                    top_base_state = base_np
                    top_labels = labels_np
                    top_base_labels = base_labels_np
                    top_delta_prob_neub = delta_prob_neub_np
                    top_delta_prob_gliob = delta_prob_gliob_np
                    top_source_state = source_np
                    top_local_effect = local_np
                    top_row_for_plot = pd.Series(row)
    if dose_rows:
        summary_df = pd.concat([summary_df, pd.DataFrame(dose_rows)], ignore_index=True)

    summary_csv = out_dir / "gene_perturbation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    local_effect_df = pd.DataFrame(local_effect_rows)
    local_effect_csv = out_dir / "local_perturbation_effect.csv"
    local_effect_df.to_csv(local_effect_csv, index=False)
    fate_enrichment_df = pd.DataFrame(fate_enrichment)
    if not fate_enrichment_df.empty:
        fate_enrichment_df["rank_sum_fdr"] = p_adjust_bh(fate_enrichment_df["rank_sum_pvalue"].astype(float).tolist())
        fate_enrichment_df["rank_sum_fdr_pert_greater"] = p_adjust_bh(
            fate_enrichment_df["rank_sum_pvalue_pert_greater"].astype(float).tolist()
        )
        fate_enrichment_df["rank_sum_fdr_pert_less"] = p_adjust_bh(
            fate_enrichment_df["rank_sum_pvalue_pert_less"].astype(float).tolist()
        )
    fate_enrichment_csv = out_dir / "fate_enrichment_regvelo_style.csv"
    fate_enrichment_df.to_csv(fate_enrichment_csv, index=False)
    sensitivity = build_critical_sensitivity(summary_df)
    sensitivity_csv = out_dir / "critical_window_sensitivity.csv"
    sensitivity.to_csv(sensitivity_csv, index=False)
    marker_df = build_literature_annotation(list(ranked["gene"].astype(str)) if not ranked.empty else screen_genes, adata, args.raw_data)
    marker_csv = out_dir / "literature_gene_annotation.csv"
    marker_df.to_csv(marker_csv, index=False)
    missing_csv = out_dir / "missing_literature_markers.csv"
    marker_df[(marker_df["gene"].isin(DEFAULT_MARKERS)) & (~marker_df["can_be_directly_perturbed"])].to_csv(missing_csv, index=False)

    figures: dict[str, dict[str, str]] = {}
    figures["baseline_real_generated_keyframes"] = plot_baseline_keyframes(
        adata=adata,
        args=args,
        controls=controls,
        source_pairs=source_pairs,
        terminal_input=float(terminal_input),
        spatial_dim=spatial_dim,
        classifier=classifier,
        label_map=label_map,
        out_dir=out_dir,
    )
    figures["local_perturbation_effect_spatial_map"] = plot_local_effect_spatial(
        top_row=top_row_for_plot,
        source_state=top_source_state,
        local_effect=top_local_effect,
        out_dir=out_dir,
    )
    figures["critical_sensitivity_by_time"] = plot_critical_sensitivity(sensitivity, out_dir)
    figures["gene_volcano_GlioB"] = plot_volcano(ranked, "GlioB", out_dir)
    figures["gene_volcano_NeuB"] = plot_volcano(ranked, "NeuB", out_dir)
    figures["top_gene_dose_response"] = plot_top_gene_dose_response(
        summary_df,
        dose_genes,
        out_dir,
        float(args.critical_time),
        args,
    )
    figures["top_gene_fate_shift_bar"] = plot_top_gene_fate_shift_bar(ranked, out_dir)
    figures["top_gene_fate_heatmap"] = plot_top_gene_fate_heatmap(ranked, out_dir)
    figures["terminal_spatial_top_perturbation"] = plot_terminal_spatial(
        adata=adata,
        args=args,
        top_row=top_row_for_plot,
        top_state=top_state,
        base_state=top_base_state,
        top_labels=top_labels,
        terminal_input=float(terminal_input),
        out_dir=out_dir,
    )
    figures["slice_terminal_counterfactual"] = plot_slice_terminal_counterfactual(
        adata=adata,
        args=args,
        top_row=top_row_for_plot,
        top_state=top_state,
        base_state=top_base_state,
        top_labels=top_labels,
        base_labels=top_base_labels,
        terminal_input=float(terminal_input),
        out_dir=out_dir,
    )
    figures["slice_fate_switch_map"] = plot_slice_fate_switch_map(
        adata=adata,
        args=args,
        top_row=top_row_for_plot,
        top_state=top_state,
        base_state=top_base_state,
        top_labels=top_labels,
        base_labels=top_base_labels,
        terminal_input=float(terminal_input),
        out_dir=out_dir,
    )
    figures["slice_delta_P_NeuB"] = plot_slice_delta_probability(
        adata=adata,
        args=args,
        top_row=top_row_for_plot,
        top_state=top_state,
        delta_prob=top_delta_prob_neub,
        label="NeuB",
        terminal_input=float(terminal_input),
        out_dir=out_dir,
    )
    figures["slice_delta_P_GlioB"] = plot_slice_delta_probability(
        adata=adata,
        args=args,
        top_row=top_row_for_plot,
        top_state=top_state,
        delta_prob=top_delta_prob_gliob,
        label="GlioB",
        terminal_input=float(terminal_input),
        out_dir=out_dir,
    )
    per_source_rows: list[dict[str, Any]] = []
    for source_time in sorted(top_by_source_time):
        payload = top_by_source_time[source_time]
        tag = f"E{source_time:g}".replace(".", "p")
        row_for_plot = payload["row"]
        per_source_rows.append({"selected_for": "per_source_screen_top", **row_for_plot.to_dict()})
        figures[f"slice_terminal_counterfactual_{tag}"] = plot_slice_terminal_counterfactual(
            adata=adata,
            args=args,
            top_row=row_for_plot,
            top_state=payload["state"],
            base_state=payload["base_state"],
            top_labels=payload["labels"],
            base_labels=payload["base_labels"],
            terminal_input=float(terminal_input),
            out_dir=out_dir,
            output_name=f"slice_by_source_time/slice_terminal_counterfactual_{tag}",
        )
        figures[f"slice_fate_switch_map_{tag}"] = plot_slice_fate_switch_map(
            adata=adata,
            args=args,
            top_row=row_for_plot,
            top_state=payload["state"],
            base_state=payload["base_state"],
            top_labels=payload["labels"],
            base_labels=payload["base_labels"],
            terminal_input=float(terminal_input),
            out_dir=out_dir,
            output_name=f"slice_by_source_time/slice_fate_switch_map_{tag}",
        )
        figures[f"slice_delta_P_NeuB_{tag}"] = plot_slice_delta_probability(
            adata=adata,
            args=args,
            top_row=row_for_plot,
            top_state=payload["state"],
            delta_prob=payload["delta_prob_neub"],
            label="NeuB",
            terminal_input=float(terminal_input),
            out_dir=out_dir,
            output_name=f"slice_by_source_time/slice_delta_P_NeuB_{tag}",
        )
        figures[f"slice_delta_P_GlioB_{tag}"] = plot_slice_delta_probability(
            adata=adata,
            args=args,
            top_row=row_for_plot,
            top_state=payload["state"],
            delta_prob=payload["delta_prob_gliob"],
            label="GlioB",
            terminal_input=float(terminal_input),
            out_dir=out_dir,
            output_name=f"slice_by_source_time/slice_delta_P_GlioB_{tag}",
        )
    per_source_top_csv = out_dir / "per_source_time_top_perturbations.csv"
    pd.DataFrame(per_source_rows).to_csv(per_source_top_csv, index=False)
    figures["potential_response_top_genes"] = plot_potential_response(summary_df, dose_genes, out_dir)
    figures["literature_marker_availability"] = plot_marker_availability(marker_df, out_dir)
    figures["overview_montage"] = plot_overview_montage(figures, out_dir)

    metadata = {
        "status": "DONE",
        "definition": (
            "Dorsal midbrain gene-expression-level in silico perturbation. "
            "Generated source-time latent states are decoded to expression to estimate a gene-specific GAT direction; "
            "the default perturbation is a mild z+x shift in gene latent space, followed by rollout to E16.5 "
            "with the trained hybrid/stVCR model. All responses are model-inferred, not experimental perturbations."
        ),
        "run_dir": str(run_dir),
        "trained_h5ad": str(summary["trained_h5ad"]),
        "model_path": str(summary["model_path"]),
        "classifier_path": str(classifier_path),
        "gat_checkpoint": str(gat.checkpoint_path),
        "gat_stvcr_latent_alignment": gat.alignment_metrics,
        "output_dir": str(out_dir),
        "device": str(device),
        "perturbation_mode": str(args.perturbation_mode),
        "latent_add_x": float(args.latent_add_x),
        "latent_add_max_x": float(args.latent_add_max_x),
        "latent_direction_expression_dose": float(args.latent_direction_expression_dose),
        "source_times": [float(v) for v in args.source_times],
        "source_times_input": [float(v) for v in source_inputs],
        "critical_window": [13.5, 13.7],
        "critical_time_used_for_prescreen_and_dose_response": float(dose_response_source[0]),
        "terminal_time": float(args.terminal_time),
        "terminal_time_input": float(terminal_input),
        "source_label": str(args.source_label),
        "screen_genes": screen_genes,
        "dose_response_genes": dose_genes,
        "screen_doses": screen_doses,
        "dose_response_doses": [float(v) for v in args.dose_response_doses],
        "seeds": seeds,
        "max_source_cells": int(args.max_source_cells),
        "control_definition": "dose=0 decode-expression then encode-GAT control at each source time and seed",
        "files": {
            "gene_perturbation_summary_csv": str(summary_csv),
            "gene_prescreen_scores_csv": str(prescreen_csv),
            "gene_screen_ranked_csv": str(ranked_csv),
            "local_perturbation_effect_csv": str(local_effect_csv),
            "fate_enrichment_regvelo_style_csv": str(fate_enrichment_csv),
            "critical_window_sensitivity_csv": str(sensitivity_csv),
            "literature_gene_annotation_csv": str(marker_csv),
            "missing_literature_markers_csv": str(missing_csv),
            "per_source_time_top_perturbations_csv": str(per_source_top_csv),
        },
        "figures": figures,
    }
    summary_json = out_dir / "perturbation_summary.json"
    legacy_summary_json = out_dir / "perturbation_run_summary.json"
    metadata["files"]["perturbation_summary_json"] = str(summary_json)
    metadata["files"]["legacy_perturbation_run_summary_json"] = str(legacy_summary_json)
    summary_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    legacy_summary_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    interpretation_path = write_interpretation(
        out_dir=out_dir,
        sensitivity=sensitivity,
        ranked=ranked,
        local_effect=local_effect_df,
        fate_enrichment=fate_enrichment_df,
        marker_df=marker_df,
        summary=metadata,
    )
    metadata["files"]["regvelo_style_interpretation_md"] = str(interpretation_path)
    metadata["files"]["legacy_interpretation_md"] = str(out_dir / "PERTURBATION_INTERPRETATION.md")
    summary_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    legacy_summary_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

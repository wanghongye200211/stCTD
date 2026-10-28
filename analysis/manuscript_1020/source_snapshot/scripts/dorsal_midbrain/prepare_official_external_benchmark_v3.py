#!/usr/bin/env python
"""Prepare the official-only external benchmark package.

The v2 package mixed project-local adapted baselines with public method names.
This v3 package does not do that.  It prepares official package inputs, command
manifests, and audit tables.  Official results are allowed into the main
benchmark only after they are produced by the listed official package/API/CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "result_figures" / "model_comparison_official_external_benchmark_v3"
INPUT_DIR = OUT_DIR / "inputs"
RUN_DIR = OUT_DIR / "official_runs"
PYTHON = Path("/opt/anaconda3/envs/stVCR/bin/python")

CACHE_ENV = {
    "HOME": "/private/tmp/stvcr-official-home",
    "MPLCONFIGDIR": "/private/tmp/mpl-stvcr",
    "NUMBA_CACHE_DIR": "/private/tmp/numba-stvcr",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    observed_h5ad: Path
    spapot_run: Path
    spapot_prediction_glob: str
    time_obs_key: str = "time"
    time_input_key: str = "time_input"
    annotation_key: str = "Annotation"
    detailed_key: str = ""


DATASETS = [
    DatasetSpec(
        dataset_id="mouse_dorsal_midbrain",
        display_name="Mouse dorsal midbrain",
        observed_h5ad=ROOT
        / "SpaPOT_runs"
        / "dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign"
        / "stvcr_notebook_trained.h5ad",
        spapot_run=ROOT / "SpaPOT_runs" / "dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign",
        spapot_prediction_glob="predictions/predict_*.h5ad",
        detailed_key="",
    ),
    DatasetSpec(
        dataset_id="zesta_10_12_18",
        display_name="Zebrafish ZESTA 10/12/18 hpf",
        observed_h5ad=ROOT
        / "SpaPOT_runs"
        / "zesta_10_12_18_gat10_h96_192_hybrid_prealigned_nolabel_e1000_s512"
        / "stvcr_notebook_hybrid_trained.h5ad",
        spapot_run=ROOT / "SpaPOT_runs" / "zesta_10_12_18_gat10_h96_192_hybrid_prealigned_nolabel_e1000_s512",
        spapot_prediction_glob="predictions_nolabel/predict_*.h5ad",
        detailed_key="DetailedAnnotation",
    ),
]


OFFICIAL_METHODS = [
    {
        "method": "SpaPOT",
        "implementation_class": "project_method_existing_run",
        "official_package_or_repo": "SpaPOT/stVCR project code",
        "official_source": "",
        "role": "main model",
        "target_aware": False,
        "fully_generative": True,
        "alignment_only": False,
        "main_benchmark_candidate": True,
        "requires_heavy_training": False,
        "manuscript_boundary": "Use as the proposed method; not an external baseline.",
    },
    {
        "method": "POT official solver",
        "implementation_class": "official_python_package",
        "official_package_or_repo": "POT",
        "official_source": "https://pythonot.github.io/; https://github.com/PythonOT/POT",
        "role": "external OT baseline",
        "target_aware": True,
        "fully_generative": False,
        "alignment_only": False,
        "main_benchmark_candidate": True,
        "requires_heavy_training": False,
        "manuscript_boundary": "Official POT solver with a documented target-aware barycentric protocol; not a fully generative trajectory model.",
    },
    {
        "method": "moscot official",
        "implementation_class": "official_python_package",
        "official_package_or_repo": "moscot",
        "official_source": "https://moscot.readthedocs.io/en/latest/; https://github.com/theislab/moscot",
        "role": "external temporal OT baseline",
        "target_aware": True,
        "fully_generative": False,
        "alignment_only": False,
        "main_benchmark_candidate": True,
        "requires_heavy_training": False,
        "manuscript_boundary": "Official moscot TemporalProblem baseline; target slices are used during OT solving.",
    },
    {
        "method": "Spateo official",
        "implementation_class": "official_python_package",
        "official_package_or_repo": "spateo-release",
        "official_source": "https://spateo-release.readthedocs.io/en/latest/; https://github.com/aristoteleo/spateo-release",
        "role": "external spatial alignment baseline",
        "target_aware": True,
        "fully_generative": False,
        "alignment_only": True,
        "main_benchmark_candidate": True,
        "requires_heavy_training": False,
        "manuscript_boundary": "Official Spateo spatial alignment/reconstruction comparison; do not describe as a temporal generative model.",
    },
    {
        "method": "TrajectoryNet official",
        "implementation_class": "official_python_package",
        "official_package_or_repo": "TrajectoryNet",
        "official_source": "https://github.com/KrishnaswamyLab/TrajectoryNet; https://arxiv.org/abs/2002.04461",
        "role": "external neural trajectory baseline",
        "target_aware": False,
        "fully_generative": True,
        "alignment_only": False,
        "main_benchmark_candidate": True,
        "requires_heavy_training": True,
        "manuscript_boundary": "Official TrajectoryNet package/CLI on the prepared joint state input; not the previous local VelocityNet baseline.",
    },
    {
        "method": "PRESCIENT official",
        "implementation_class": "official_python_package",
        "official_package_or_repo": "prescient",
        "official_source": "https://github.com/gifford-lab/prescient; https://pypi.org/project/prescient/",
        "role": "external potential-gradient trajectory baseline",
        "target_aware": False,
        "fully_generative": True,
        "alignment_only": False,
        "main_benchmark_candidate": True,
        "requires_heavy_training": True,
        "manuscript_boundary": "Official PRESCIENT process/train/simulate code path; neutral growth weights are explicitly documented when no growth metadata are available. PRESCIENT predictions are expression/PCA trajectories, so spatial metrics are not applicable.",
    },
]


EXCLUDED_LEGACY_METHODS = [
    {
        "method": "TrajectoryNet-like",
        "previous_source": "Local PyTorch VelocityNet + Sinkhorn loss",
        "exclusion_reason": "Not the official TrajectoryNet implementation; excluded from v3 main benchmark.",
    },
    {
        "method": "PRESCIENT-adapted",
        "previous_source": "Local potential-gradient baseline",
        "exclusion_reason": "Not the official PRESCIENT software; excluded from v3 main benchmark.",
    },
    {
        "method": "POT-OT proxy / WOT-style OT proxy",
        "previous_source": "Project-local OT proxy protocol",
        "exclusion_reason": "Replaced by a manifest row for the official POT solver protocol.",
    },
    {
        "method": "moscot-WOT proxy",
        "previous_source": "Project-local WOT-style proxy",
        "exclusion_reason": "Replaced by official moscot TemporalProblem protocol.",
    },
]


def configure_cache_env() -> None:
    for key, value in CACHE_ENV.items():
        os.environ.setdefault(key, value)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def shell_env_prefix() -> str:
    return " ".join(f"{key}={value}" for key, value in CACHE_ENV.items())


def as_dense(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def safe_name(name: object) -> str:
    text = str(name)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "feature"


def state_matrix(adata) -> np.ndarray:
    if "X_spatial_aligned" not in adata.obsm:
        raise KeyError("Missing adata.obsm['X_spatial_aligned']")
    if "X_gene_input" in adata.obsm:
        gene = np.asarray(adata.obsm["X_gene_input"], dtype=np.float32)
    elif "X_gene_prior_gatae_10d" in adata.obsm:
        gene = np.asarray(adata.obsm["X_gene_prior_gatae_10d"], dtype=np.float32)
    else:
        gene = as_dense(adata.X).astype(np.float32)
    spatial = np.asarray(adata.obsm["X_spatial_aligned"], dtype=np.float32)
    return np.hstack([spatial, gene]).astype(np.float32)


def standardized_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = state.mean(axis=0, keepdims=True)
    std = state.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((state - mean) / std).astype(np.float32), mean.ravel(), std.ravel()


def consecutive_time_labels(times: Iterable[float]) -> tuple[np.ndarray, dict[float, int]]:
    unique_times = sorted(float(t) for t in pd.unique(pd.Series(times)))
    mapping = {time_value: idx for idx, time_value in enumerate(unique_times)}
    labels = np.array([mapping[float(t)] for t in times], dtype=np.int64)
    return labels, mapping


def load_scanpy():
    import scanpy as sc

    return sc


def write_trajectorynet_input(spec: DatasetSpec, adata, state: np.ndarray) -> dict[str, object]:
    labels, mapping = consecutive_time_labels(adata.obs[spec.time_input_key].to_numpy())
    state_z, mean, std = standardized_state(state)
    out_dir = INPUT_DIR / spec.dataset_id / "trajectorynet"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{spec.dataset_id}_state_trajectorynet.npz"
    np.savez_compressed(
        npz_path,
        state=state_z.astype(np.float32),
        sample_labels=labels.astype(np.int64),
        original_time_input=adata.obs[spec.time_input_key].to_numpy(dtype=float),
        original_time=adata.obs[spec.time_obs_key].to_numpy(dtype=float),
        state_mean=mean.astype(np.float32),
        state_std=std.astype(np.float32),
    )
    meta_path = out_dir / f"{spec.dataset_id}_trajectorynet_time_mapping.json"
    meta_path.write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2) + "\n")
    return {
        "dataset_id": spec.dataset_id,
        "method": "TrajectoryNet official",
        "input_type": "npz",
        "path": str(npz_path),
        "n_cells": int(state_z.shape[0]),
        "n_features": int(state_z.shape[1]),
        "time_mapping": str(meta_path),
        "notes": "Official TrajectoryNet CustomData input: npz with sample_labels and embedding array named 'state'.",
    }


def write_prescient_inputs(spec: DatasetSpec, adata, write_expression: bool) -> list[dict[str, object]]:
    out_dir = INPUT_DIR / spec.dataset_id / "prescient"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, mapping = consecutive_time_labels(adata.obs[spec.time_input_key].to_numpy())
    annotation = adata.obs[spec.annotation_key].astype(str).to_numpy()
    meta = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "time_index": labels.astype(int),
            "original_time_input": adata.obs[spec.time_input_key].to_numpy(dtype=float),
            "original_time": adata.obs[spec.time_obs_key].to_numpy(dtype=float),
            "Annotation": annotation,
        }
    )
    if spec.detailed_key and spec.detailed_key in adata.obs:
        meta[spec.detailed_key] = adata.obs[spec.detailed_key].astype(str).to_numpy()
    meta_path = out_dir / f"{spec.dataset_id}_metadata.csv"
    meta.to_csv(meta_path, index=False)

    growth_path = out_dir / f"{spec.dataset_id}_neutral_growth.pt"
    source_label = min(mapping.values())
    source_n = int(np.sum(labels == source_label))
    import torch

    torch.save({"w": {source_label: np.zeros(source_n, dtype=np.float32)}}, growth_path)

    rows = [
        {
            "dataset_id": spec.dataset_id,
            "method": "PRESCIENT official",
            "input_type": "metadata_csv",
            "path": str(meta_path),
            "n_cells": int(adata.n_obs),
            "n_features": "",
            "time_mapping": json.dumps({str(k): v for k, v in mapping.items()}),
            "notes": "Metadata for official PRESCIENT process_data.",
        },
        {
            "dataset_id": spec.dataset_id,
            "method": "PRESCIENT official",
            "input_type": "neutral_growth_pt",
            "path": str(growth_path),
            "n_cells": source_n,
            "n_features": "",
            "time_mapping": "",
            "notes": "Neutral log-growth weights because no proliferation/growth metadata were identified in the curated benchmark h5ad.",
        },
    ]

    expression_path = out_dir / f"{spec.dataset_id}_expression.csv"
    if write_expression:
        expr = as_dense(adata.X).astype(np.float32)
        var_names = [safe_name(v) for v in adata.var_names]
        expr_df = pd.DataFrame(expr, index=adata.obs_names.astype(str), columns=var_names)
        expr_df.to_csv(expression_path, float_format="%.6g")
        expression_note = "Expression CSV written for official PRESCIENT process_data."
    elif expression_path.exists():
        expression_note = "Expression CSV already exists for official PRESCIENT process_data."
    else:
        expression_note = "Expression CSV not written in this pass; rerun with --write-prescient-expression to materialize it."
    rows.append(
        {
            "dataset_id": spec.dataset_id,
            "method": "PRESCIENT official",
            "input_type": "expression_csv",
            "path": str(expression_path),
            "n_cells": int(adata.n_obs),
            "n_features": int(adata.n_vars),
            "time_mapping": "",
            "notes": expression_note,
        }
    )
    return rows


def write_dataset_summary(spec: DatasetSpec, adata, state: np.ndarray) -> dict[str, object]:
    labels, mapping = consecutive_time_labels(adata.obs[spec.time_input_key].to_numpy())
    counts = pd.Series(labels).value_counts().sort_index()
    return {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "observed_h5ad": str(spec.observed_h5ad),
        "n_cells": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "state_dim": int(state.shape[1]),
        "time_mapping_original_to_consecutive": json.dumps({str(k): v for k, v in mapping.items()}),
        "consecutive_time_counts": json.dumps({str(int(k)): int(v) for k, v in counts.items()}),
        "spapot_run": str(spec.spapot_run),
        "spapot_prediction_glob": str(spec.spapot_run / spec.spapot_prediction_glob),
    }


def existing_spapot_predictions(spec: DatasetSpec) -> str:
    return ";".join(str(path) for path in sorted(spec.spapot_run.glob(spec.spapot_prediction_glob)))


def build_method_manifest(dataset_summaries: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary in dataset_summaries:
        spec = next(item for item in DATASETS if item.dataset_id == summary["dataset_id"])
        for method in OFFICIAL_METHODS:
            if method["method"] == "SpaPOT":
                official_status = "completed_existing_project_run"
                run_path = str(spec.spapot_run)
                prediction_path = existing_spapot_predictions(spec)
                qualified = True
                exclusion = ""
            elif (RUN_DIR / spec.dataset_id / safe_name(method["method"]) / "official_metrics_by_time.csv").exists():
                completed_dir = RUN_DIR / spec.dataset_id / safe_name(method["method"])
                official_status = "completed_official_run"
                run_path = str(completed_dir)
                prediction_path = ";".join(str(path) for path in sorted((completed_dir / "predictions").glob("*.h5ad")))
                qualified = True
                exclusion = ""
            else:
                official_status = "pending_official_run"
                run_path = str(RUN_DIR / spec.dataset_id / safe_name(method["method"]))
                prediction_path = ""
                qualified = False
                exclusion = "Official output not generated yet in v3; exclude from final figures until completed."
            rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "dataset": spec.display_name,
                    "method": method["method"],
                    "role": method["role"],
                    "implementation_class": method["implementation_class"],
                    "official_package_or_repo": method["official_package_or_repo"],
                    "official_source": method["official_source"],
                    "target_aware": method["target_aware"],
                    "fully_generative": method["fully_generative"],
                    "alignment_only": method["alignment_only"],
                    "main_benchmark_candidate": method["main_benchmark_candidate"],
                    "official_status": official_status,
                    "qualified_for_main_figure_now": qualified,
                    "exclusion_reason": exclusion,
                    "run_path": run_path,
                    "prediction_path": prediction_path,
                    "manuscript_boundary": method["manuscript_boundary"],
                }
            )
    return pd.DataFrame(rows)


def build_training_scale_audit(dataset_summaries: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary in dataset_summaries:
        dataset_id = str(summary["dataset_id"])
        pot_done = (RUN_DIR / dataset_id / "POT_official_solver" / "official_metrics_by_time.csv").exists()
        moscot_done = (RUN_DIR / dataset_id / "moscot_official" / "official_metrics_by_time.csv").exists()
        spateo_done = (RUN_DIR / dataset_id / "Spateo_official" / "official_metrics_by_time.csv").exists()
        trajectorynet_done = (RUN_DIR / dataset_id / "TrajectoryNet_official" / "official_metrics_by_time.csv").exists()
        prescient_done = (RUN_DIR / dataset_id / "PRESCIENT_official" / "official_metrics_by_time.csv").exists()
        prescient_processed = (RUN_DIR / dataset_id / "PRESCIENT_official" / "processed" / "data.pt").exists()
        rows.extend(
            [
                {
                    "dataset_id": dataset_id,
                    "method": "SpaPOT",
                    "source_path": str(next(item.spapot_run for item in DATASETS if item.dataset_id == dataset_id)),
                    "training_unit": "epoch",
                    "epochs_or_niters": 1000,
                    "batch_size_or_sample_size": 512,
                    "hidden_dim": "dataset-specific existing run",
                    "n_layers": "existing run",
                    "latent_dim": 10,
                    "qualified_for_main": True,
                    "exclusion_reason": "",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "TrajectoryNet official",
                    "source_path": str(RUN_DIR / dataset_id / "TrajectoryNet_official"),
                    "training_unit": "iteration",
                    "epochs_or_niters": 400,
                    "batch_size_or_sample_size": 512,
                    "hidden_dim": 128,
                    "n_layers": "128-128 CNF hidden dims, 1 CNF block, solver=rk4, step_size=0.1 for MPS float32 compatibility",
                    "latent_dim": int(summary["state_dim"]),
                    "qualified_for_main": trajectorynet_done,
                    "exclusion_reason": ""
                    if trajectorynet_done
                    else "Pending official TrajectoryNet run; command manifest uses niters=400, batch_size=512, hidden=128, and the official training loop.",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "PRESCIENT official",
                    "source_path": str(RUN_DIR / dataset_id / "PRESCIENT_official"),
                    "training_unit": "epoch",
                    "epochs_or_niters": "pretrain=400;train=400",
                    "batch_size_or_sample_size": "train_batch=0.25 fraction",
                    "hidden_dim": 128,
                    "n_layers": "2, official prescient train_model; CPU fallback because the official OT loss path uses float64 tensors unsupported by MPS",
                    "latent_dim": 10,
                    "qualified_for_main": prescient_done,
                    "exclusion_reason": ""
                    if prescient_done
                    else "Pending official PRESCIENT process/train/simulate run; no 120/5/1 epoch result is allowed.",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "POT official solver",
                    "source_path": str(RUN_DIR / dataset_id / "POT_official_solver"),
                    "training_unit": "none",
                    "epochs_or_niters": "",
                    "batch_size_or_sample_size": "all cells per source/target pair unless an explicitly audited solver limit is needed",
                    "hidden_dim": "",
                    "n_layers": "",
                    "latent_dim": int(summary["state_dim"]),
                    "qualified_for_main": pot_done,
                    "exclusion_reason": "" if pot_done else "Pending official POT-solver output.",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "moscot official",
                    "source_path": str(RUN_DIR / dataset_id / "moscot_official"),
                    "training_unit": "none",
                    "epochs_or_niters": "",
                    "batch_size_or_sample_size": "all cells; solver batch_size only if required by memory",
                    "hidden_dim": "",
                    "n_layers": "",
                    "latent_dim": int(summary["state_dim"]),
                    "qualified_for_main": moscot_done,
                    "exclusion_reason": "" if moscot_done else "Pending official moscot output.",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "Spateo official",
                    "source_path": str(RUN_DIR / dataset_id / "Spateo_official"),
                    "training_unit": "alignment",
                    "epochs_or_niters": "numItermax=200;numItermaxEmd=100000",
                    "batch_size_or_sample_size": "all cells unless official alignment fails memory audit",
                    "hidden_dim": "",
                    "n_layers": "",
                    "latent_dim": int(summary["state_dim"]),
                    "qualified_for_main": spateo_done,
                    "exclusion_reason": "" if spateo_done else "Pending official Spateo alignment output; label as alignment-only.",
                },
            ]
        )
        if prescient_processed:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": "PRESCIENT official process_data",
                    "source_path": str(RUN_DIR / dataset_id / "PRESCIENT_official" / "processed" / "data.pt"),
                    "training_unit": "preprocessing",
                    "epochs_or_niters": "",
                    "batch_size_or_sample_size": "all cells",
                    "hidden_dim": "",
                    "n_layers": "",
                    "latent_dim": 10,
                    "qualified_for_main": False,
                    "exclusion_reason": (
                        "Preprocessing audit row only; the benchmark method is tracked in the PRESCIENT official row."
                        if prescient_done
                        else "Official PRESCIENT preprocessing completed, but train_model/simulate are still pending."
                    ),
                }
            )
    for excluded in EXCLUDED_LEGACY_METHODS:
        rows.append(
            {
                "dataset_id": "legacy_v2",
                "method": excluded["method"],
                "source_path": excluded["previous_source"],
                "training_unit": "legacy",
                "epochs_or_niters": "",
                "batch_size_or_sample_size": "",
                "hidden_dim": "",
                "n_layers": "",
                "latent_dim": "",
                "qualified_for_main": False,
                "exclusion_reason": excluded["exclusion_reason"],
            }
        )
    return pd.DataFrame(rows)


def command_rows(dataset_summaries: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    env = shell_env_prefix()
    for summary in dataset_summaries:
        dataset_id = str(summary["dataset_id"])
        tnet_input = INPUT_DIR / dataset_id / "trajectorynet" / f"{dataset_id}_state_trajectorynet.npz"
        pres_dir = INPUT_DIR / dataset_id / "prescient"
        pres_expr = pres_dir / f"{dataset_id}_expression.csv"
        pres_meta = pres_dir / f"{dataset_id}_metadata.csv"
        pres_growth = pres_dir / f"{dataset_id}_neutral_growth.pt"
        pres_processed = RUN_DIR / dataset_id / "PRESCIENT_official" / "processed"
        pres_model = RUN_DIR / dataset_id / "PRESCIENT_official" / "model"
        tnet_run = RUN_DIR / dataset_id / "TrajectoryNet_official"
        pot_done = (RUN_DIR / dataset_id / "POT_official_solver" / "official_metrics_by_time.csv").exists()
        moscot_done = (RUN_DIR / dataset_id / "moscot_official" / "official_metrics_by_time.csv").exists()
        spateo_done = (RUN_DIR / dataset_id / "Spateo_official" / "official_metrics_by_time.csv").exists()
        tnet_train_done = (tnet_run / "checkpt-0400.pth").exists() or (tnet_run / "checkpt.pth").exists()
        tnet_materialized_done = (tnet_run / "official_metrics_by_time.csv").exists()
        pres_process_done = (pres_processed / "data.pt").exists()
        pres_train_done = any(pres_model.glob("*/seed_0/train.epoch_000400.pt"))
        pres_sim_done = (RUN_DIR / dataset_id / "PRESCIENT_official" / "official_metrics_by_time.csv").exists()
        rows.extend(
            [
                {
                    "dataset_id": dataset_id,
                    "method": "POT official solver",
                    "stage": "run",
                    "command": f"{env} {PYTHON} scripts/dorsal_midbrain/run_official_external_baselines_v3.py --dataset-id {dataset_id} --methods pot",
                    "official_entrypoint": "ot.sinkhorn / ot.da mapping functions from POT",
                    "scale_gate": "Use all cells unless a memory failure is logged in the method audit.",
                    "status": "completed_artifact_present" if pot_done else "not_run_by_prepare_script",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "moscot official",
                    "stage": "run",
                    "command": f"{env} {PYTHON} scripts/dorsal_midbrain/run_official_external_baselines_v3.py --dataset-id {dataset_id} --methods moscot",
                    "official_entrypoint": "moscot.problems.time.TemporalProblem",
                    "scale_gate": "Use all cells and official solver; do not replace with local WOT proxy.",
                    "status": "completed_artifact_present" if moscot_done else "not_run_by_prepare_script",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "Spateo official",
                    "stage": "run",
                    "command": f"{env} {PYTHON} scripts/dorsal_midbrain/run_official_external_baselines_v3.py --dataset-id {dataset_id} --methods spateo",
                    "official_entrypoint": "spateo.alignment.paste_align or morpho_align",
                    "scale_gate": "Alignment-only target-aware baseline; not a trajectory generator.",
                    "status": "completed_artifact_present" if spateo_done else "not_run_by_prepare_script",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "TrajectoryNet official",
                    "stage": "train",
                    "command": (
                        f"{env} {PYTHON} scripts/dorsal_midbrain/run_trajectorynet_official_device_v3.py --force-device mps "
                        f"--dataset {tnet_input} --embedding_name state --max_dim {summary['state_dim']} "
                        f"--dims 128-128 --num_blocks 1 --niters 400 --batch_size 512 --test_batch_size 512 "
                        f"--viz_batch_size 512 --solver rk4 --step_size 0.1 --save_freq 400 --save {tnet_run}"
                    ),
                    "official_entrypoint": "TrajectoryNet official dataset/model/loss/train loop via run_trajectorynet_official_device_v3.py",
                    "scale_gate": "niters=400, batch_size=512, hidden=128, official rk4 solver for MPS; no local VelocityNet baseline is accepted.",
                    "status": "completed_artifact_present" if tnet_train_done else "not_run_by_prepare_script",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "TrajectoryNet official",
                    "stage": "materialize_predictions",
                    "command": (
                        f"{env} {PYTHON} scripts/dorsal_midbrain/materialize_official_neural_predictions_v3.py "
                        f"--dataset-id {dataset_id} --method trajectorynet --seed 0 --device mps "
                        f"--trajectorynet-solver rk4 --trajectorynet-step-size 0.1"
                    ),
                    "official_entrypoint": "TrajectoryNet official model classes/checkpoint generation",
                    "scale_gate": "Run only after official checkpt.pth exists; writes prediction h5ad and metrics.",
                    "status": "completed_artifact_present" if tnet_materialized_done else "not_run_by_prepare_script",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "PRESCIENT official",
                    "stage": "process_data",
                    "command": (
                        f"{env} {PYTHON} -m prescient process_data -d {pres_expr} -m {pres_meta} "
                        f"--tp_col time_index --celltype_col Annotation --num_pcs 10 --growth_path {pres_growth} -o {pres_processed}/"
                    ),
                    "official_entrypoint": "python -m prescient process_data",
                    "scale_gate": "Official PRESCIENT data preprocessing; expression CSV must exist.",
                    "status": "completed_artifact_present" if pres_process_done else "not_run_by_prepare_script",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "PRESCIENT official",
                    "stage": "train_model",
                    "command": (
                        f"{env} {PYTHON} -m prescient train_model "
                        f"-i {pres_processed}/data.pt --out_dir {pres_model} "
                        f"--seed 0 --weight_name neutral --k_dim 128 --layers 2 "
                        f"--pretrain_epochs 400 --train_epochs 400 --train_batch 0.25 --save 100 --no-cuda"
                    ),
                    "official_entrypoint": "python -m prescient train_model",
                    "scale_gate": "pretrain=400 and train=400, k_dim=128, layers=2; official CPU fallback is used because PRESCIENT's official OT loss path creates float64 tensors that MPS cannot execute.",
                    "status": "completed_artifact_present" if pres_train_done else "not_run_by_prepare_script",
                },
                {
                    "dataset_id": dataset_id,
                    "method": "PRESCIENT official",
                    "stage": "simulate",
                    "command": (
                        f"{env} {PYTHON} scripts/dorsal_midbrain/materialize_official_neural_predictions_v3.py "
                        f"--dataset-id {dataset_id} --method prescient --seed 0 --prescient-epoch 000400 --device cpu"
                    ),
                    "official_entrypoint": "prescient.simulate.simulate via official package",
                    "scale_gate": "Run only after official train.epoch_000400.pt exists; spatial metrics are not applicable for expression-only PRESCIENT.",
                    "status": "completed_artifact_present" if pres_sim_done else "not_run_by_prepare_script",
                },
            ]
        )
    return pd.DataFrame(rows)


def write_policy_markdown() -> None:
    text = """# Official-only external benchmark policy

本输出包替代 `model_comparison_external_benchmark_v2` 作为外部模型正式基准的准备包。v2 中的 `TrajectoryNet-like`、`PRESCIENT-adapted` 和 WOT/proxy 类结果不能再进入主图。

## 纳入规则

- 主外部模型必须来自官方 Python 包、官方仓库代码或官方命令入口。
- SpaPOT 是本研究方法，可复用现有 `e1000_s512` 正式 run。
- TrajectoryNet 必须用 `python -m TrajectoryNet.main/eval` 或同一官方包内部模型代码，不能使用本地 `VelocityNet`。
- PRESCIENT 必须用官方 `prescient process_data/train_model` 与官方 simulate 代码，不能使用本地 potential-gradient surrogate。
- POT 必须调用官方 `POT` 包 solver/API；若使用目标切片构建 barycentric map，必须标注为 target-aware solver baseline。
- moscot 必须调用官方 `moscot.problems.time.TemporalProblem`，不能写成 local WOT proxy。
- Spateo 必须调用官方 `spateo.alignment`，并标注为空间 alignment/reconstruction baseline，不写成 fully generative trajectory model。

## 训练尺度

- TrajectoryNet official: `niters=400`, `batch_size=512`, hidden dims `128-128`，使用官方 TrajectoryNet data/model/loss/train loop；本机可用时显式使用 MPS。MPS 下使用官方支持的 `rk4` solver 和 `step_size=0.1`，避免 `dopri5`/torchdiffeq 的 float64 张量与 MPS 不兼容。
- PRESCIENT official: `pretrain_epochs=400`, `train_epochs=400`, `k_dim=128`, `layers=2`；使用官方 `prescient train_model` / `prescient.train.run`。已尝试 MPS，但官方 OT loss 路径会产生 MPS 不支持的 float64 张量；正式结果使用官方 CPU 路径，不私自改 dtype、loss 或模型。
- 不允许 `e1/e3/e5`、smoke、debug、timing 或 120 epoch 本地快速模型进入主 benchmark。
- 官方 heavy-run 完成前，对应方法只保留在 manifest/command audit，不生成正式 main figure 数值。

## 论文表述边界

- `TrajectoryNet-like` 和 `PRESCIENT-adapted` 只能作为被排除的 legacy local baselines 记录，不能引用为官方模型结果。
- POT/moscot/Spateo 都使用目标切片信息，不能写成 fully generative baseline。
- 如果 PRESCIENT 使用 neutral growth weights，应在 Methods/Supplement 说明原因：当前两个 h5ad benchmark 没有可审计的 proliferation/growth metadata。
"""
    (OUT_DIR / "OFFICIAL_METHOD_POLICY_zh.md").write_text(text)


def write_readme(dataset_summaries: pd.DataFrame) -> None:
    summary_columns = ["dataset_id", "display_name", "n_cells", "n_vars", "state_dim", "observed_h5ad"]
    summary_table = ["| " + " | ".join(summary_columns) + " |"]
    summary_table.append("| " + " | ".join(["---"] * len(summary_columns)) + " |")
    for _, row in dataset_summaries[summary_columns].iterrows():
        summary_table.append("| " + " | ".join(str(row[col]) for col in summary_columns) + " |")
    text = f"""# model_comparison_official_external_benchmark_v3

This package is the official-only external benchmark package for the two formal datasets.

Generated files:

- `official_environment_audit.csv/json`: local `stVCR` conda environment package and import audit.
- `official_method_manifest.csv`: official method status and manuscript wording boundaries.
- `official_training_scale_audit.csv`: scale gates for main benchmark inclusion.
- `official_baseline_command_manifest.csv`: exact official package/API/CLI commands used or prepared for reruns.
- `official_input_manifest.csv`: prepared input files for official TrajectoryNet and PRESCIENT.
- `excluded_legacy_local_baselines.csv`: local adapted/proxy methods excluded from v3.
- `dataset_summary.csv`: observed data and state dimensions for the two datasets.
- `OFFICIAL_METHOD_POLICY_zh.md`: Chinese policy note for manuscript-safe wording.
- `official_completed_raw_metrics.csv`: time-resolved source data for completed official methods.
- `official_completed_summary_excluding_source.csv`: per-dataset method means excluding the source time.
- `official_completed_normalized_score_heatmap.svg/pdf/png/tiff`: normalized score overview.
- `official_completed_prediction_effect_panel.svg/pdf/png/tiff`: target-time prediction-effect panel.
- `official_completed_raw_metric_dotplot.svg/pdf/png/tiff`: per-dataset raw metric panel.
- `official_completed_prediction_effect_source.csv` and `official_completed_per_dataset_raw_metric_table.csv`: source data for the added panels.
- `official_slice_reconstruction_<dataset_id>.svg/pdf/png/tiff`: spatial slice reconstruction panels.
- `official_slice_reconstruction_points.csv` and `official_slice_reconstruction_source_manifest.csv`: point-level source data and source paths for slice reconstruction panels.
- `official_completed_deliverable_audit.csv/json`: persistent completion checks for methods, source data, figures, and SVG integrity.

Current datasets:

{os.linesep.join(summary_table)}

Important: this prepare script does not fabricate official results. A method only enters the formal completed tables when `official_method_manifest.csv` marks `qualified_for_main_figure_now=True` and the corresponding prediction/metric source files exist.
"""
    (OUT_DIR / "README.md").write_text(text)


def run_environment_audit() -> None:
    env = os.environ.copy()
    env.update(CACHE_ENV)
    cmd = [str(PYTHON), str(ROOT / "scripts" / "dorsal_midbrain" / "audit_official_baseline_environment.py")]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def write_tables(
    input_rows: list[dict[str, object]],
    dataset_summaries: list[dict[str, object]],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_df = pd.DataFrame(dataset_summaries)
    dataset_df.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    input_df = pd.DataFrame(input_rows)
    input_df.to_csv(OUT_DIR / "official_input_manifest.csv", index=False)
    manifest = build_method_manifest(dataset_summaries)
    manifest.to_csv(OUT_DIR / "official_method_manifest.csv", index=False)
    scale = build_training_scale_audit(dataset_summaries)
    scale.to_csv(OUT_DIR / "official_training_scale_audit.csv", index=False)
    commands = command_rows(dataset_summaries)
    commands.to_csv(OUT_DIR / "official_baseline_command_manifest.csv", index=False)
    pd.DataFrame(EXCLUDED_LEGACY_METHODS).to_csv(OUT_DIR / "excluded_legacy_local_baselines.csv", index=False)
    write_policy_markdown()
    write_readme(dataset_df)

    validation_rows = [
        {
            "check": "legacy_local_methods_not_qualified_for_main",
            "passed": not manifest["method"].isin(["TrajectoryNet-like", "PRESCIENT-adapted"]).any(),
            "details": "official_method_manifest contains only official external methods plus SpaPOT.",
        },
        {
            "check": "heavy_neural_scale_gate",
            "passed": bool(
                (
                    (scale["method"].eq("TrajectoryNet official") & scale["epochs_or_niters"].eq(400)).any()
                    and (
                        scale["method"].eq("PRESCIENT official")
                        & scale["epochs_or_niters"].astype(str).str.contains("pretrain=400;train=400", regex=False)
                    ).any()
                )
            ),
            "details": "TrajectoryNet and PRESCIENT command scale gates are at formal-run scale.",
        },
        {
            "check": "target_aware_boundaries_recorded",
            "passed": bool(manifest.loc[manifest["method"].isin(["POT official solver", "moscot official", "Spateo official"]), "target_aware"].all()),
            "details": "POT, moscot, and Spateo are marked target-aware and not fully generative.",
        },
    ]
    pd.DataFrame(validation_rows).to_csv(OUT_DIR / "validation_audit.csv", index=False)
    (OUT_DIR / "validation_audit.json").write_text(json.dumps(validation_rows, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-prescient-expression",
        action="store_true",
        help="Materialize full expression CSV files required by official PRESCIENT process_data.",
    )
    parser.add_argument(
        "--skip-environment-audit",
        action="store_true",
        help="Skip rerunning the package import/pip-check environment audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cache_env()
    sc = load_scanpy()

    input_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    for spec in DATASETS:
        adata = sc.read_h5ad(spec.observed_h5ad)
        state = state_matrix(adata)
        dataset_summaries.append(write_dataset_summary(spec, adata, state))
        input_rows.append(write_trajectorynet_input(spec, adata, state))
        input_rows.extend(write_prescient_inputs(spec, adata, args.write_prescient_expression))

    write_tables(input_rows, dataset_summaries)
    if not args.skip_environment_audit:
        run_environment_audit()
    print(json.dumps({"out_dir": str(OUT_DIR), "datasets": len(dataset_summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchdiffeq import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STVCR_SRC = PROJECT_ROOT / "stVCR_repo" / "src"
if str(STVCR_SRC) not in sys.path:
    sys.path.insert(0, str(STVCR_SRC))

from stvcr.training.model import stVCR_DynamicModel  # noqa: E402


PALETTE = {
    "Epidermal": "#F4A261",
    "Eye": "#E9C46A",
    "Forebrain": "#2F80ED",
    "Mesoderm": "#66C2A5",
    "Nervous System": "#9B51E0",
    "Neural Crest": "#D62728",
    "Neural Keel": "#17BECF",
    "Neural Rod": "#56B4E9",
    "Otic Vesicle": "#8D6E63",
    "Yolk Syncytial Layer": "#7F7F7F",
}


class StateClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float,
        n_hiddens: int,
        activation: str,
    ) -> None:
        super().__init__()
        if activation == "silu":
            act: type[nn.Module] = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            act = nn.ReLU
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(n_hiddens)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify and visualize ZESTA stVCR generated intermediate frames.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "SpaPOT_runs" / "zesta_10_12_18_original_stvcr_hybrid_prealigned_nolabel_e1000_s512",
    )
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--frame-step", type=float, default=0.5)
    parser.add_argument("--source-time", type=float, default=10.0)
    parser.add_argument("--end-time", type=float, default=18.0)
    parser.add_argument("--ode-method", default="rk4")
    parser.add_argument("--ode-step-size", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-hiddens", type=int, default=6)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--classifier-hidden-dim", type=int, default=128)
    parser.add_argument("--classifier-n-hiddens", type=int, default=2)
    parser.add_argument("--classifier-activation", default="relu", choices=["relu", "silu", "gelu"])
    parser.add_argument("--classifier-epochs", type=int, default=450)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--spatial-key", default="X_spatial_aligned")
    parser.add_argument("--gene-key", default="X_gene_input")
    parser.add_argument("--time-key", default="time_input")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--output-name", default="classified_dynamic_0p5")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dynamics_model(
    model_path: Path,
    gene_dim: int,
    spatial_dim: int,
    device: torch.device,
    hidden_dim: int,
    n_hiddens: int,
    activation: str,
) -> stVCR_DynamicModel:
    payload = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(payload, stVCR_DynamicModel):
        return payload.to(device).eval()
    model = stVCR_DynamicModel(
        in_out_gene_dim=gene_dim,
        spatial_dim=spatial_dim,
        hidden_dim=hidden_dim,
        n_hiddens=n_hiddens,
        activation=activation,
        use_gene=True,
        use_spatial=True,
        velocity_parameterization="hybrid",
    ).to(device)
    model.load_state_dict(payload)
    model.eval()
    return model


def integrate_model(
    model: stVCR_DynamicModel,
    state0: np.ndarray,
    source_time_input: float,
    target_time_input: float,
    device: torch.device,
    method: str,
    step_size: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isclose(source_time_input, target_time_input):
        return state0.copy(), np.ones(state0.shape[0], dtype=np.float32)
    states = []
    weights = []
    for start in range(0, state0.shape[0], chunk_size):
        stop = min(start + chunk_size, state0.shape[0])
        chunk = torch.as_tensor(state0[start:stop], dtype=torch.float32, device=device)
        logw0 = torch.zeros(chunk.shape[0], 1, dtype=torch.float32, device=device)
        t = torch.tensor([source_time_input, target_time_input], dtype=torch.float32, device=device)
        with torch.enable_grad():
            z_time, logw_time = odeint(
                model,
                (chunk, logw0),
                t=t,
                atol=1e-5,
                rtol=1e-5,
                method=method,
                options={"step_size": float(step_size)} if method in {"rk4", "euler", "midpoint"} else {},
            )
        states.append(z_time[-1].detach().cpu().numpy().astype(np.float32))
        weights.append(torch.exp(logw_time[-1]).reshape(-1).detach().cpu().numpy().astype(np.float32))
        del chunk, logw0, z_time, logw_time
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(states), np.concatenate(weights)


def make_features(spatial: np.ndarray, gene: np.ndarray, time_input: np.ndarray) -> np.ndarray:
    return np.hstack([spatial, gene, np.asarray(time_input, dtype=np.float32).reshape(-1, 1)]).astype(np.float32)


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[StateClassifier, StandardScaler, LabelEncoder, dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    scaler = StandardScaler()
    x = scaler.fit_transform(features).astype(np.float32)

    train_idx, val_idx = train_test_split(
        np.arange(x.shape[0]),
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    class_counts = np.bincount(y[train_idx], minlength=len(encoder.classes_)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    model = StateClassifier(
        x.shape[1],
        args.classifier_hidden_dim,
        len(encoder.classes_),
        args.classifier_dropout,
        args.classifier_n_hiddens,
        args.classifier_activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))

    x_train = torch.as_tensor(x[train_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y[train_idx], dtype=torch.long, device=device)
    x_val = torch.as_tensor(x[val_idx], dtype=torch.float32, device=device)
    y_val_np = y[val_idx]

    best_state = None
    best_bal_acc = -np.inf
    trace = []
    for epoch in range(int(args.classifier_epochs)):
        perm = rng.permutation(x_train.shape[0])
        model.train()
        batch_losses = []
        for start in range(0, len(perm), 512):
            batch = perm[start : start + 512]
            logits = model(x_train[batch])
            loss = loss_fn(logits, y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        if epoch % 25 == 0 or epoch == int(args.classifier_epochs) - 1:
            model.eval()
            with torch.no_grad():
                pred = model(x_val).argmax(dim=1).detach().cpu().numpy()
            bal_acc = float(balanced_accuracy_score(y_val_np, pred))
            acc = float(accuracy_score(y_val_np, pred))
            row = {"epoch": int(epoch), "loss": float(np.mean(batch_losses)), "val_acc": acc, "val_bal_acc": bal_acc}
            trace.append(row)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.as_tensor(x, dtype=torch.float32, device=device)).argmax(dim=1).detach().cpu().numpy()
    confusion = confusion_matrix(y, train_pred, labels=np.arange(len(encoder.classes_)))
    summary = {
        "classes": [str(value) for value in encoder.classes_],
        "feature_set": "[aligned spatial, gene latent, time_input]",
        "hidden_dim": int(args.classifier_hidden_dim),
        "n_hiddens": int(args.classifier_n_hiddens),
        "activation": str(args.classifier_activation),
        "dropout": float(args.classifier_dropout),
        "epochs": int(args.classifier_epochs),
        "train_accuracy": float(accuracy_score(y, train_pred)),
        "train_balanced_accuracy": float(balanced_accuracy_score(y, train_pred)),
        "best_val_balanced_accuracy": float(best_bal_acc),
        "trace": trace,
        "confusion_matrix": confusion.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": [str(value) for value in encoder.classes_],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "summary": summary,
        },
        output_dir / "state_classifier.pt",
    )
    pd.DataFrame(confusion, index=encoder.classes_, columns=encoder.classes_).to_csv(output_dir / "classifier_confusion.csv")
    return model, scaler, encoder, summary


def predict_labels(
    classifier: StateClassifier,
    scaler: StandardScaler,
    encoder: LabelEncoder,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x = scaler.transform(features).astype(np.float32)
    labels = []
    probs = []
    classifier.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], 4096):
            batch = torch.as_tensor(x[start : start + 4096], dtype=torch.float32, device=device)
            logits = classifier(batch)
            prob = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(prob)
            labels.append(encoder.inverse_transform(prob.argmax(axis=1)))
    return np.concatenate(labels), np.vstack(probs)


def label_colors(labels: np.ndarray, categories: list[str]) -> list[str]:
    fallback = plt.get_cmap("tab20")
    category_colors = {
        category: PALETTE.get(category, matplotlib.colors.to_hex(fallback(i % 20)))
        for i, category in enumerate(categories)
    }
    return [category_colors[str(label)] for label in labels]


def add_legend(fig: plt.Figure, categories: list[str]) -> None:
    fallback = plt.get_cmap("tab20")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=PALETTE.get(category, matplotlib.colors.to_hex(fallback(i % 20))),
            label=category,
            markersize=6,
        )
        for i, category in enumerate(categories)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False, fontsize=8)


def style_axis(ax: plt.Axes, xy_limits: tuple[float, float, float, float], dashed: bool) -> None:
    xmin, xmax, ymin, ymax = xy_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        if dashed:
            spine.set_linestyle((0, (3, 3)))


def plot_keyframes(
    real_frames: list[dict[str, object]],
    categories: list[str],
    path: Path,
    title: str,
    point_size: float,
    xy_limits: tuple[float, float, float, float],
) -> None:
    fig, axes = plt.subplots(1, len(real_frames), figsize=(3.8 * len(real_frames), 4.0), squeeze=False)
    for ax, frame in zip(axes.ravel(), real_frames):
        xy = frame["xy"]
        labels = frame["labels"]
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=label_colors(labels, categories), linewidths=0, alpha=0.9)
        ax.set_title(f"real {frame['time']:g} hpf", fontsize=10)
        style_axis(ax, xy_limits, dashed=False)
    fig.suptitle(title, fontsize=13)
    add_legend(fig, categories)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_generated_timeline(
    frames: list[dict[str, object]],
    categories: list[str],
    path: Path,
    title: str,
    point_size: float,
    xy_limits: tuple[float, float, float, float],
    ncols: int,
) -> None:
    nrows = int(np.ceil(len(frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.65 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, frame in zip(axes.ravel(), frames):
        xy = frame["xy"]
        labels = frame["labels"]
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=label_colors(labels, categories), linewidths=0, alpha=0.9)
        ax.set_title(f"generated {frame['time']:g} hpf", fontsize=8)
        style_axis(ax, xy_limits, dashed=True)
    fig.suptitle(title, fontsize=13)
    add_legend(fig, categories)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_alternating_timeline(
    frames: list[dict[str, object]],
    real_by_time: dict[float, dict[str, object]],
    categories: list[str],
    path: Path,
    title: str,
    point_size: float,
    xy_limits: tuple[float, float, float, float],
    ncols: int,
) -> None:
    nrows = int(np.ceil(len(frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.65 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, gen_frame in zip(axes.ravel(), frames):
        time_value = float(gen_frame["time"])
        frame = real_by_time.get(time_value, gen_frame)
        is_generated = time_value not in real_by_time
        xy = frame["xy"]
        labels = frame["labels"]
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=label_colors(labels, categories), linewidths=0, alpha=0.9)
        prefix = "generated" if is_generated else "real"
        ax.set_title(f"{prefix} {time_value:g} hpf", fontsize=8)
        style_axis(ax, xy_limits, dashed=is_generated)
    fig.suptitle(title, fontsize=13)
    add_legend(fig, categories)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def make_gif(
    frames: list[dict[str, object]],
    categories: list[str],
    path: Path,
    point_size: float,
    xy_limits: tuple[float, float, float, float],
) -> None:
    import imageio.v2 as imageio

    temp_dir = path.parent / "gif_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    png_paths = []
    for index, frame in enumerate(frames):
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
        xy = frame["xy"]
        labels = frame["labels"]
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size + 1.0, c=label_colors(labels, categories), linewidths=0, alpha=0.9)
        ax.set_title(f"generated {frame['time']:g} hpf", fontsize=12)
        style_axis(ax, xy_limits, dashed=False)
        fig.tight_layout()
        png_path = temp_dir / f"frame_{index:03d}.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(png_path)
    images = [imageio.imread(path_) for path_ in png_paths]
    imageio.mimsave(path, images, duration=0.35)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = run_dir / args.output_name
    frame_dir = output_dir / "generated_h5ad"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    trained_h5ad = run_dir / "stvcr_notebook_hybrid_trained.h5ad"
    model_path = run_dir / "save_model" / "stvcr_notebook_hybrid_model.pth"
    adata = ad.read_h5ad(str(trained_h5ad))

    spatial = np.asarray(adata.obsm[args.spatial_key], dtype=np.float32)
    gene = np.asarray(adata.obsm[args.gene_key], dtype=np.float32)
    time_input = np.asarray(adata.obs[args.time_key], dtype=np.float32)
    labels = adata.obs[args.annotation_key].astype(str).to_numpy()
    features = make_features(spatial, gene, time_input)
    categories = sorted(str(value) for value in pd.unique(labels))

    classifier, scaler, encoder, classifier_summary = train_classifier(features, labels, args, device, output_dir)

    source_mask = np.isclose(np.asarray(adata.obs["time"].astype(float), dtype=np.float32), args.source_time)
    source_time_input = float(np.median(time_input[source_mask]))
    initial_state = np.hstack([spatial[source_mask], gene[source_mask]]).astype(np.float32)
    source_obs_names = adata.obs_names[source_mask].astype(str).to_numpy()
    source_annotations = labels[source_mask]
    spatial_dim = spatial.shape[1]
    gene_dim = gene.shape[1]
    dynamics_model = load_dynamics_model(
        model_path,
        gene_dim,
        spatial_dim,
        device,
        args.hidden_dim,
        args.n_hiddens,
        args.activation,
    )

    real_frames = []
    for time_value in sorted(float(value) for value in np.unique(adata.obs["time"].astype(float))):
        mask = np.isclose(np.asarray(adata.obs["time"].astype(float), dtype=np.float32), time_value)
        real_frames.append({"time": time_value, "xy": spatial[mask], "labels": labels[mask]})
    real_by_time = {float(frame["time"]): frame for frame in real_frames}

    frame_times = np.round(np.arange(args.source_time, args.end_time + 1e-8, args.frame_step), 6)
    rng = np.random.default_rng(args.seed)
    generated_frames = []
    count_rows = []
    for raw_time in frame_times:
        target_time_input = float(raw_time - args.source_time + source_time_input)
        state_all, weights_all = integrate_model(
            dynamics_model,
            initial_state,
            source_time_input,
            target_time_input,
            device,
            args.ode_method,
            args.ode_step_size,
            args.chunk_size,
        )
        weights = np.maximum(weights_all.astype(np.float64), 1e-8)
        probs = weights / weights.sum()
        n_frame = max(1, int(round(float(weights.sum()))))
        chosen = np.arange(state_all.shape[0]) if n_frame == state_all.shape[0] else rng.choice(state_all.shape[0], size=n_frame, replace=True, p=probs)
        state = state_all[chosen]
        xy = state[:, :spatial_dim]
        z = state[:, spatial_dim:]
        frame_features = make_features(xy, z, np.full(xy.shape[0], target_time_input, dtype=np.float32))
        pred_labels, pred_probs = predict_labels(classifier, scaler, encoder, frame_features, device)
        generated_frames.append({"time": float(raw_time), "time_input": target_time_input, "xy": xy, "labels": pred_labels})

        pred = ad.AnnData(X=sp.csr_matrix((len(pred_labels), adata.n_vars), dtype=np.float32), var=adata.var.copy())
        pred.obs_names = [f"zesta_gen_{str(raw_time).replace('.', 'p')}_{i}" for i in range(pred.n_obs)]
        pred.obs["time"] = float(raw_time)
        pred.obs[args.time_key] = target_time_input
        pred.obs["predicted_annotation"] = pred_labels
        pred.obs["predicted_confidence"] = pred_probs.max(axis=1)
        pred.obs["source_index"] = chosen.astype(int)
        pred.obs["source_cell_id"] = source_obs_names[chosen]
        pred.obs["source_annotation"] = source_annotations[chosen]
        pred.obs["source_weight"] = weights_all[chosen]
        copy_counter: dict[int, int] = {}
        copy_ids = []
        for source_index in chosen.astype(int):
            copy_id = copy_counter.get(source_index, 0)
            copy_ids.append(copy_id)
            copy_counter[source_index] = copy_id + 1
        pred.obs["source_copy_id"] = copy_ids
        pred.obsm[args.spatial_key] = xy
        pred.obsm[args.gene_key] = z
        pred.write_h5ad(str(frame_dir / f"generated_{str(raw_time).replace('.', 'p')}.h5ad"))

        counts = pd.Series(pred_labels).value_counts()
        count_row = {"time": float(raw_time), "time_input": target_time_input, "n_generated": int(len(pred_labels)), "weight_sum": float(weights.sum())}
        for category in categories:
            count_row[category] = int(counts.get(category, 0))
        count_rows.append(count_row)

    all_xy = np.vstack([spatial] + [frame["xy"] for frame in generated_frames])
    xpad = 0.04 * max(float(np.ptp(all_xy[:, 0])), 1e-8)
    ypad = 0.04 * max(float(np.ptp(all_xy[:, 1])), 1e-8)
    xy_limits = (
        float(all_xy[:, 0].min() - xpad),
        float(all_xy[:, 0].max() + xpad),
        float(all_xy[:, 1].min() - ypad),
        float(all_xy[:, 1].max() + ypad),
    )

    real_png = output_dir / "zesta_real_keyframes_annotation.png"
    generated_png = output_dir / "zesta_generated_annotation_timeline_0p5.png"
    alternating_png = output_dir / "zesta_real_generated_alternating_annotation_0p5.png"
    gif_path = output_dir / "zesta_generated_annotation_0p5.gif"
    plot_keyframes(real_frames, categories, real_png, "ZESTA real keyframes: Annotation", args.point_size, xy_limits)
    plot_generated_timeline(generated_frames, categories, generated_png, "ZESTA generated frames: classifier Annotation", args.point_size, xy_limits, ncols=6)
    plot_alternating_timeline(
        generated_frames,
        real_by_time,
        categories,
        alternating_png,
        "ZESTA real/generated alternating timeline",
        args.point_size,
        xy_limits,
        ncols=6,
    )
    make_gif(generated_frames, categories, gif_path, args.point_size, xy_limits)

    counts_csv = output_dir / "generated_annotation_counts_0p5.csv"
    pd.DataFrame(count_rows).to_csv(counts_csv, index=False)
    summary = {
        "status": "DONE",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "trained_h5ad": str(trained_h5ad),
        "model_path": str(model_path),
        "classifier": classifier_summary,
        "frame_step": float(args.frame_step),
        "frame_times": [float(value) for value in frame_times],
        "categories": categories,
        "figures": {
            "real_keyframes": str(real_png),
            "generated_timeline": str(generated_png),
            "alternating_timeline": str(alternating_png),
            "gif": str(gif_path),
            "counts_csv": str(counts_csv),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

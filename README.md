# stCTD

**Spatiotemporal Critical Transition Decipherer** reconstructs continuous
population dynamics from time-series spatial transcriptomic snapshots. It
couples spatial transport, potential-driven molecular dynamics, and source–sink
population growth.

[Project website](https://wanghongye200211.github.io/stCTDcover/) ·
[Code](https://github.com/wanghongye200211/stCTD) ·
[Website source](https://github.com/wanghongye200211/stCTDcover)

This repository contains the model, graph-informed embedding helpers, and a
training and reconstruction-evaluation entry point. Start with a prepared AnnData
file or the small synthetic example below. Data, trained weights, and manuscript
figures are not bundled.

## Manuscript analysis and reproducibility

The [1020 analysis package](analysis/manuscript_1020/README.md) contains frozen
score inputs and lightweight commands to reproduce all 256 rows of Supplementary
Tables S4–S6 and the six biological slope-change posteriors used for Figure S34.
These commands reuse the existing analysis implementation and require no model
training. Historical STE, gene-sensitivity, perturbation, and benchmark scripts
are also included as source snapshots with SHA-256 provenance.

Saved-score replay is currently the supported paper-analysis entry point.
End-to-end recreation from raw expression data remains incomplete: prepared
datasets, fitted weights, coordinate conversions, and selected-run dependencies
are not distributed here. Several score definitions differ between the main
results and supplementary panels; do not treat them as interchangeable. See the
[submission audit and result-to-code map](docs/submission_audit_20261028.md)
for exact coverage, Methods differences, and the remaining release requirements.

## Install

Python 3.9 or later is required. From a terminal:

```bash
git clone https://github.com/wanghongye200211/stCTD.git
cd stCTD
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml`; `requirements.txt` installs the
same editable package. The CLI accepts `--device cpu`, `cuda`, `mps`, or `auto`.
Automatic selection prefers available MPS, then CUDA, then CPU.

## Quick example

Create a synthetic input, then run a small CPU training and evaluation job:

```bash
python examples/make_demo_data.py --output runs/demo_input.h5ad
python examples/train_stctd.py \
  --input-h5ad runs/demo_input.h5ad \
  --out-dir runs/demo \
  --device cpu \
  --epochs 3 --sample-size 16 \
  --hidden-dim 16 --n-hidden 2 \
  --ode-step-size 0.5 --no-sample-growth --quiet
```

The demo checks the software workflow; its synthetic latent coordinates are not
GAT embeddings, and it is not a biological benchmark. `--epochs` controls the
dynamics model. Evaluation subsequently trains its own cell-type classifier,
which can continue after the dynamics epochs finish.

Use a separate `--out-dir` for each run because existing run outputs are replaced.

## Prepare your data

The entry point uses precomputed coordinates and molecular embeddings. It does
not run tissue alignment or train an embedding automatically.

| AnnData field | Default key | Required content |
| --- | --- | --- |
| `obsm` | `X_spatial_input` | Finite aligned coordinates, shape `(n_cells, 2)` |
| `obsm` | `X_gene_input` | Finite molecular latent coordinates, shape `(n_cells, latent_dim)` |
| `obs` | `time_input` | Numeric model time, with at least two distinct stages |
| `obs` | `time` | Numeric experimental time corresponding to each model stage |
| `obs` | `Annotation` | Cell or spot labels, with no missing values |
| `layers` | `lognorm` | Finite normalized gene expression; falls back to `X` if absent |

Cells or spots must use a common gene order and a common molecular latent space.
The model time and experimental time should each identify the same distinct
stages. For example, experimental days `[10, 12, 14]` can correspond to model
times `[0, 1, 2]`. Units are determined by this mapping.

Supply different field names with `--spatial-key`, `--latent-key`, `--time-key`,
`--raw-time-key`, `--annotation-key`, and `--expression-layer-key`. Latent width
is inferred from the input. Spatial and latent features are standardized
separately over the supplied observations; standardized spatial coordinates are
weighted by `3.0`. `--no-scale-features` disables both standardization and spatial
weighting.

To reconstruct gene expression, supply `--decoder-checkpoint` with the matching
embedding decoder, latent convention, and gene order. Without a decoder, the
prediction's `X` is a **zero placeholder**; spatial coordinates, molecular
embeddings, and label predictions remain available. Each prediction records this
choice in `uns['stctd_expression_source']`.

## Train on a prepared dataset

```bash
python examples/train_stctd.py \
  --input-h5ad path/to/prepared_spatial_latent.h5ad \
  --out-dir runs/stctd \
  --device auto
```

The default `stctd_fullgrid` objective propagates particles forward from the
first observed stage and backward from the last stage, matching both trajectories
to the observed time grid. Its action term has a fixed coefficient of one;
`--alpha-exp` and `--alpha-gro` control its molecular and growth contributions.
Analysis-specific loss weights should be selected for each dataset.

| Option | Meaning |
| --- | --- |
| `--lambda-hj FLOAT` | First-order Hamilton–Jacobi penalty for potential dynamics |
| `--velocity-parameterization potential\|vector` | Manuscript model or direct molecular-velocity ablation |
| `--loss-mode endpoint` | Endpoint objective for controlled ablations |
| `--lambda-action FLOAT` | Action coefficient in endpoint mode |
| `--lambda-ssp FLOAT` | Spatial-coherence penalty in endpoint mode |
| `--use-cell-type-prior` | Cell-type matching prior in endpoint mode |
| `--require-obs-value KEY VALUE` | Assert that every observation has the requested value; does not filter cells |
| `--baseline-summary path/to/summary.json` | Add an existing run to the comparison table; repeat for several runs |

The vector ablation cannot use the Hamilton–Jacobi penalty. SSP and cell-type
priors require endpoint mode. See all options with:

```bash
python examples/train_stctd.py --help
```

## Read the outputs

```text
runs/stctd/
├── config.json                    Input fields, model and training configuration
├── summary.json                   Training status and evaluation results
├── checkpoints/
│   ├── model.pt                   Weights, model configuration and feature scaler
│   └── training_trace.jsonl        Loss values and matching terms
├── predictions/*.h5ad             Reconstructed snapshots
├── comparisons/*.png              Observed versus reconstructed spatial plots
├── metrics.csv                    Label and spatial reconstruction metrics
├── mass_diagnostics.csv           Particle-weight growth diagnostics
└── stctd_vs_baseline_metrics.csv   Current run and optional baseline comparisons
```

Evaluation also saves its classifier checkpoint and label mapping. The default
full-grid evaluation uses a classifier over spatial position, latent state, and
time. Endpoint evaluation uses a latent-state classifier.

Predicted snapshots are resampled to the **observed cell count** at each stage.
Their cell counts are therefore not independent growth predictions. Inspect
`mass_diagnostics.csv` for predicted particle mass. The comparison table reports
reconstruction metrics at the fitted stages, including the initial stage; these
are not held-out forecasting results.

## Python API

```python
from stctd import ModelConfig, STCTDModel
from stctd.train import train_stctd_model
from stctd.evaluate import evaluate_stctd_model

model = STCTDModel(ModelConfig())
```

Restore a dynamics checkpoint:

```python
import torch
from stctd import ModelConfig, STCTDModel

checkpoint = torch.load("runs/stctd/checkpoints/model.pt", map_location="cpu", weights_only=True)
model = STCTDModel(ModelConfig(**checkpoint["model_config"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

Model inputs must use the saved feature scaler in `checkpoint['scaler']` and the
same model-time convention used during training.

## Model and interpretation

For joint state `x = [s, z]`, the three coupled fields are:

```text
ds/dt      = v_theta(s, z, t)
dz/dt      = -grad_z Phi_theta(s, z, t)
d log w/dt = g_theta(s, z, t)

partial_t rho + div_s(rho v_theta)
              - div_z(rho grad_z Phi_theta) = g_theta rho
```

The spatial and molecular-potential branches each use six hidden layers of
width 128; the growth branch uses three hidden layers. Defaults are ReLU
activations, Adam optimization, bidirectional full-series fitting, and fixed-step
RK4 integration. The vector parameterization is an explicit ablation. See
[model provenance](docs/model_provenance.md) for checkpoint architecture and
historical naming.

The method reconstructs population dynamics between independently sampled
specimens; it does not track the same physical cells over time. Transition
readouts along a trajectory fitted to the full observed series support
retrospective localization. Prospective early warning requires past-only fitting,
past-only preprocessing, and held-out future validation. The training entry point
exports reconstruction results. The separate [manuscript analysis package](analysis/manuscript_1020/README.md)
replays saved-score statistics and provides historical analysis source for
inspection; it does not turn the demo into an end-to-end paper reproduction.

## Repository and checks

| Path | Purpose |
| --- | --- |
| `src/stctd/` | Model, data loading, training, integration, and evaluation |
| `src/embedding/` | Graph-informed molecular embedding and decoder helpers |
| `examples/` | Prepared-data training CLI and synthetic input generator |
| `tests/` | Architecture, data contract, and CLI regression checks |
| `docs/model_provenance.md` | Model and historical checkpoint provenance |
| `tools/audit_stctd_equivalence.py` | Optional comparison with a historical checkpoint |

Run the regression checks after installation:

```bash
python -m unittest discover -s tests -v
```

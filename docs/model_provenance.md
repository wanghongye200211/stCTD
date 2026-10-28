# Model and Figure Provenance

This document separates the canonical stCTD implementation from historical run
and checkpoint identifiers. Renderer names do not define the model; the model
configuration, checkpoint, and run summary do.

## Canonical stCTD Model

For joint state `x = [s, z]`, stCTD uses:

```text
ds/dt      = v_theta(s, z, t)
dz/dt      = -grad_z Phi_theta(s, z, t)
d log w/dt = g_theta(s, z, t)
```

The corresponding population equation is:

```text
partial_t rho + div_s(rho v_theta)
              - div_z(rho grad_z Phi_theta) = g_theta rho
```

The manuscript model therefore has three coupled fields:

```text
spatial branch:             direct spatial-velocity MLP
molecular branch:           scalar-potential MLP with -grad_z Phi output
source-sink growth branch:  scalar growth MLP
```

The direct molecular-vector option in the code is an ablation, not the
manuscript model.

## Architecture

The reference checkpoints use:

```text
spatial_dim: 2
latent_dim: 10
hidden_dim: 128
activation: ReLU
spatial branch: 6 hidden layers
molecular-potential branch: 6 hidden layers
growth branch: 3 hidden layers
```

Observed checkpoint parameter shapes are:

```text
spatial_velocity_net:
  hidden linears: (128, 13), then five (128, 128), output (2, 128)
molecular-potential net:
  hidden linears: (128, 13), then five (128, 128), scalar output (1, 128)
growth_rate_net:
  hidden linears: (128, 13), (128, 128), (128, 128), output (1, 128)
```

The growth branch intentionally remains three layers deep even when the spatial
and molecular-potential branches use six hidden layers.

The 28 October 2026 audit checked this architecture against the current 1020
submission manuscript and supplementary information. The supplement already
states 6/6/3; the current main Methods no longer contains the older all-six-layer
statement. See the [submission consistency audit](submission_audit_20261028.md)
for the optional explicit main-text sentence and the separate analysis-version
differences. Network agreement does not establish full paper-result reproducibility.

## Training Configuration

The public defaults follow the manuscript implementation:

```text
feature preprocessing: standardize spatial and molecular features separately
spatial weight: 3.0 after standardization
training direction: bidirectional across the full observed time series
epochs: 1000
optimizer: Adam
learning rate: 1e-3
weight decay: 1e-5
gradient clipping: disabled
ODE solver: fixed-step RK4
default RK4 step size: 0.25
```

The matching, dynamic, physical, and optional spatial loss weights are
analysis-specific. The physical term is a first-order Hamilton-Jacobi constraint
on the molecular-potential field; it is not a diffusion-control HJB equation.

The default full-grid objective propagates particles from the earliest stage
forward to all observed stages and from the latest stage backward to all
observed stages. Generated and observed distributions are matched at the sampled
times while the trajectory action and source-sink behavior are regularized.

## Final Figure 5 Reference

The current final axolotl Figure 5 panels and trajectory animation were rendered
from the HJ-regularized run recorded at:

```text
result_figures/axolotl_artista_preprocessed_injured_direct_ae_hybrid_hjb/e1000_s1024_lamhjb10_samplegrowth
```

The matching animation report is:

```text
result_figures/蝾螈/trajectory_b_panel/gif_hjb_previous_fixed_dt0p1/render_report.json
```

It records 181 frames from 2.0 to 20.0 DPI at `raw_dt = 0.1`.

These paths retain historical internal names because they are immutable
provenance identifiers. Their relevant fields map to current terminology as
follows:

```text
legacy velocity_parameterization = "hybrid"  -> current "potential"
legacy lambda_hjb                          -> current lambda_hj
legacy hybrid_model.pth                    -> stCTD potential-model checkpoint
```

The recorded reference configuration includes:

```text
model_family: stCTD
model_path: save_model/hybrid_model.pth
velocity_parameterization: hybrid
lambda_hjb: 10.0
n_epochs: 1000
num_samples: [1204, 1204, 1204, 1204, 1204]
seed: 19491001
lambda_match: 400000.0
alpha_exp: 0.01
alpha_gro: 0.0002
kappa_exp: 0.02
kappa_gro: 0.1
ode_method: rk4
ode_step_size: 0.25
use_alignment: false
```

An older no-HJ rendering remains available for comparison but is not the source
for the current final Figure 5:

```text
result_figures/axolotl_artista_injured_adjacent_aligned_direct_ae_hybrid_hj_compare/nohj_lam0_e1000_s1024_samplegrowth
```

## Checkpoint Equivalence Audit

The optional audit maps the historical checkpoint weights into `STCTDModel` and
compares spatial velocity, molecular velocity, and growth on random probe states:

```bash
python tools/audit_stctd_equivalence.py \
  --reference-src path/to/reference/src \
  --checkpoint path/to/reference_hybrid_model.pth
```

Both the no-HJ checkpoint and the Figure 5 HJ-regularized checkpoint produced
zero maximum absolute differences after weight mapping for spatial velocity,
molecular velocity, and growth. The HJ term changes the training objective, not
the core potential-driven architecture.

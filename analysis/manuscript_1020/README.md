# Analysis accompanying manuscript version 1020

This package exposes existing analysis code and frozen numerical inputs. It
reproduces saved-score statistics; it does not retrain the biological models or
reconstruct their trajectories from raw data. The unresolved manuscript and
source differences are listed in the [submission audit](../../docs/submission_audit_20261028.md).

## Run the verified numerical replay

From the repository root, use a separate Python 3.9 environment (the version
used for verification) and install the analysis requirements:

```bash
python -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r analysis/manuscript_1020/requirements.txt
python analysis/manuscript_1020/replay_tables.py --output-dir runs/replay_1020_tables
python analysis/manuscript_1020/replay_bayesian.py --output-dir runs/replay_1020_bayesian
```

Use a new output directory each time. Run without Python's `-O` option: the
original numerical checks use assertions. No manuscript file is required or
modified, and no GPU or model checkpoint is required.

The first command recalculates one-sided Welch tests and within-series,
within-window-size BH adjustments, including all eligible endpoints. It verifies
33, 103 and 120 rows for Tables S4, S5 and S6 against `expected/Table_S*.csv`.
The second invokes the unchanged historical Bayesian `analyze` function for
the six biological series and checks the complete unconditional and conditional
posterior vectors, including candidate times, against saved S34 inputs.

Outputs are numerical CSVs and JSON verification records. The first script is
the existing `verify_statistics.py` with file access adapted and obsolete figure
label reports removed; [the exact diff](replay_tables_io.diff) is supplied. The
Bayesian script is an input/output wrapper around the unchanged
[score_statistics.py](score_statistics.py). No new STE or perturbation algorithm
has been inferred from manuscript prose.

On the audited macOS/NumPy environment the Bayesian module emitted floating-point
RuntimeWarnings from matrix operations, while all returned probabilities were
finite and matched the saved reference. These warnings are not suppressed here;
the unchanged historical numerical implementation has not been modernized.

## Frozen input identity and scope

`inputs/scores.csv` is copied byte-for-byte from the 20 October main-score
statistics package (SHA-256
`d54ff914f16f97019a7be22b29604ba46dd3797016f326643b501b26e2e9b1f7`).
It contains 313 nodes across nine series. The replay selects six biological
series. The five-node smooth control cannot support a four-versus-four test.

| Input | Purpose and limitation |
| --- | --- |
| `scores.csv`, `series_manifest.csv` | Exact frozen main-score inputs and their historical definitions; provenance text is retained as recorded |
| `fish_raw_components.csv` | Main zebrafish weighted-mean potential increments, spatial costs, 0.9/0.1 weights, midpoint labels |
| `mouse_components.csv` | Main mouse q90/smoothed components and 0.99/0.01 weights |
| `supplement_mean_component_variant.csv` | Separate 0.8/0.2 arithmetic-mean variant; do not use it to reproduce Tables S4–S6 |
| `simulation_0802_scores.csv` | Later 0.8/0.2 synthetic figure input; the aggregate `simulation_fig2` series retains an earlier 0.7/0.3 variant |
| `expected/` | Frozen reference outputs, including the separate synthetic Welch result; these are comparison targets, not newly generated evidence |

The table replay includes n=2 sensitivity rows for zebrafish B04 and axolotl
reaEGC, alongside n=4 primary rows. The 1020 main Fig. 4b uses n=2; main Fig. 5c
uses n=4. `reaEGC` in source files maps to the `reaRGC` label in Table S6. Test
units are correlated model-score nodes, not independent animals or embryos.
The Bayesian probabilities concern a change in score slope, not biological
critical-point probabilities. Full-series fitting, terminal cohort selection and
centered smoothing keep these analyses retrospective.

## Existing implementation snapshots

`source_snapshot/` preserves the original workspace-relative paths of historical
scripts. [source_manifest.json](source_manifest.json) records original and released
SHA-256 values; all snapshots are byte-identical. They are provided for inspection
and provenance, not advertised as standalone runnable entry points. Their original
workspace imports, caches, trained data, and local output paths remain intact.
In particular, a relocated zebrafish regional-perturbation script derives the
wrong workspace root from `parents[2]`; its classifier helper was found in
`SpaPOT_runs/zesta_scripts/`. Use the dependency map before attempting to package
that workflow. A `.pyc` file is not treated as a replacement for source code.

The snapshot includes direct-potential and historical centered-score helpers.
Importing `critical_transition_contract.py` does **not** select the manuscript's
final score automatically. Consult the dataset-specific source and audit.

## Integrity and release coverage

`source_manifest.json` maps every copied source/data file and the adapted table
script. `release_manifest.json` inventories all package files except itself.
`validation.json` records lightweight verification performed for this release.
Main-figure rendering, end-to-end raw-data processing, perturbation reruns,
large-scale training and independent biological validation were not performed.

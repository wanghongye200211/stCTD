# Submission code and Methods audit

Audit date: 28 October 2026. Public baseline: main commit
`de6181d68fcde0d8dda555b4629f64ceb80ed4c3`. GitHub returned only `main` in the
branch listing. The local publication checkout matched that commit with no
pre-existing changes. This audit does not claim to have searched every other
repository owned by the author.

The manuscript examined was the current local submission copy of
`stCTD_manuscript_revised_1020.docx`, SHA-256
`b35596a8d2a0f492f0b3408d09531dddc800c65d9b4f9d9a3cd578e443b7346e`.
The supplementary file was `stCTD_Supplementary_Information_revised.docx`, SHA-256
`a12d753118234814e54b34ee1c5f68428b937fd460101eb49f411dfd50deffe9`.
These identify the audited bytes; equivalence to a separate File Library upload
has not been established. The DOCX files were read without modification.

## Architecture finding

The historical statement that all three fields have six hidden layers does not
describe the implementation. However, it is **already absent from the current
1020 main Methods**. Supplementary Note 6, “Training and population accounting”,
already states six, six and three layers, in agreement with the public model.
The current main text says each field is an MLP without specifying its depth.

| Field | Hidden layers | Hidden width | Output | Evidence |
| --- | --- | --- | --- | --- |
| Spatial velocity | 6 | 128 | 2 | `HyperNetwork1`, `config.n_hidden=6` |
| Molecular potential | 6 | 128 | Scalar, differentiated with respect to z | `GenePotentialGradient`, `config.n_hidden=6` |
| Net growth | 3 | 128 | Scalar | `HyperNetwork2`; independent of `n_hidden` |

The default concatenated input has dimension 13 (time, two spatial coordinates,
ten latent coordinates). The growth branch has four Linear layers in total:
13→128→128→128→1. The other branches have six hidden Linear layers followed by
an output Linear layer. Hidden layers use ReLU; outputs are linear. The demo's
`--n-hidden 2` changes the spatial/potential branches only and is not a manuscript
architecture setting. See [fields.py](../src/stctd/fields.py),
[config.py](../src/stctd/config.py) and the existing architecture tests.

An optional explicit addition after “Each field is parameterized by a multilayer
perceptron” in the main Methods is:

> The spatial-velocity and molecular-potential networks each contain six hidden
> layers of 128 units, whereas the source–sink growth network contains three
> hidden layers of 128 units; all hidden layers use ReLU activations.

No architecture change or retraining is needed to fix the old wording.

The existing equivalence tool was also rerun on the located mouse and zebrafish
historical checkpoints: 32 fixed random probe states each, seed 123 and model
time 7.5. After weight mapping, spatial velocity, molecular velocity and growth
all had maximum absolute difference 0. This is an implementation-equivalence
check at those probes, not a repeat of model fitting or validation of every run.

## Public reproduction coverage

“Runnable” below refers to a tested software path, not reproduction of every
published biological result. The new analysis package is separate from the
training CLI.

| Analysis/result | Public main before this patch | This patch and remaining boundary |
| --- | --- | --- |
| Three-field dynamics, prepared-data training, reconstruction evaluation | Model, CLI, synthetic demo, tests | Existing implementation retained; 16 regression checks passed |
| GAT and decoding | Encoder/decoder helpers | Prepared biological inputs, original/recovered representation mapping and weights still need a release |
| STE from saved trajectories | Potential and integration primitives only | Original dataset-specific STE scripts and component variants are exposed for inspection; a portable trajectory-to-STE workflow is still incomplete |
| S4–S6 local-window tests | No public analysis entry | Runnable replay of all 256 table rows from frozen saved scores, including n=2 sensitivity rows |
| S34 slope-change analysis | No public analysis entry | Runnable replay of six biological posterior series using the unchanged historical Bayesian module |
| Fig. 2 synthetic benchmark | Generic demo only | Historical pitchfork generator and selected 0.8/0.2 score/reference table exposed; generic demo is not Fig. 2 |
| Fig. 3 gene ranking and perturbation | Bottom-level potential/embedding/classifier components | Historical gradient, ranking, dose screen and paired-rollout sources exposed; full data/checkpoint/affine-map bundle and exact figure-ranking linkage still required |
| Fig. 4 regional perturbation | No public high-level entry | Regional perturbation source and its existing classifier helper located; path/import relocation and selected input bundle still required |
| Fig. 5 axolotl analysis | Historical run description | STE source and saved-score inputs exposed; trained ARTISTA latent input, selected model and trajectory cache still required |
| Fig. 6 reconstruction benchmark | Generic evaluation components | Existing benchmark/metric code exposed; method-specific prediction manifests and coordinate maps still required for complete reproduction |

The public [analysis README](../analysis/manuscript_1020/README.md) gives the two
verified commands, input identities, output schema, and numerical limitations.

## Methods and result versions requiring reconciliation

These differences are not repaired by changing network depth. Keep the current
numeric records frozen until each panel's intended version has been resolved.

1. **STE formula versus actual table inputs.** Supplementary Notes 5–6 and
   S8/S13/S23–S24 describe mouse weight 0.9 and other biological profiles as
   unweighted means with 0.8/0.2 weights. The frozen inputs behind S4–S6/S34 instead
   record mouse q90 with a centered nine-node temporal moving average and
   0.99/0.01 weights; zebrafish uses growth-weighted mean absolute potential
   increments and 0.9/0.1; axolotl uses the retained q90 0.9/0.1 display score
   divided by its initial value. Temporal smoothing is distinct from subtracting
   a frame-wise potential center. The archived mean-based 0.8/0.2 components are
   a separate variant. Dividing an already transformed combined score by its
   initial value is not generally equivalent to normalizing both raw components
   separately. Thus the main Eq. (4) must not be asserted to reproduce all these
   files without the dataset-specific transformations.
2. **Zebrafish time labels.** S13 describes right endpoints; S4–S6 input tables,
   main-score files and current S34 use interval midpoints. For those current
   files, t describes [t−0.1, t+0.1] hpf. Do not shift reported events by silently
   substituting one axis convention for the other.
3. **Gene ranking.** Main Eq. (26) is a single-time median absolute expression
   gradient scaled by expression standard deviation. Note 6 instead states an
   E13.4→E13.5 change in potential-descent contribution with direction fixed at
   the later time. Both computations exist locally. The adjacent-time script
   ranks absolute differences of direction-adjusted median gradients; the
   single-time script ranks `median_abs_potential_derivative_per_sd`. These are
   distinct statistics. Fig. 3g/Table S1 need one explicit source-table and sort
   rule mapping; candidate perturbation genes are also selected separately from
   a literature-supported list. Do not claim all selections are Eq. (26) outputs.
4. **Spatial scope.** The theoretical state permits two or three spatial
   dimensions, but the current prepared-data loader explicitly accepts two.
   Describe the released CLI and reported experiments as 2D unless a separate
   3D implementation is supplied.
5. **Availability wording.** The main manuscript says documentation reproduces
   the main analyses. This was broader than the baseline repository. Even with
   this patch, the supported claim is saved-score statistical replay plus
   historical source access, with incomplete end-to-end dependencies.

The current main captions already report corrected raw P values (for example
mouse E13.7, P=0.029). Earlier audit reports containing P=0.004 are historical
and must not be used to reintroduce an already corrected discrepancy. Raw P
and BH values remain distinguishable; replay does not validate biological
criticality, prospective warning, or independence of reconstructed time nodes.

## Existing sources and exact remaining inputs

Paths below are original workspace-relative identifiers. Their source copies
are under [source_snapshot](../analysis/manuscript_1020/source_snapshot/), with
hashes in [source_manifest.json](../analysis/manuscript_1020/source_manifest.json).

| Purpose | Existing source | Inputs needed to close end-to-end reproduction |
| --- | --- | --- |
| Raw spatial cost and mean STE | `outputs/critical_curves_redefined_20260915_w08_w02/{prepare_spatial,redraw_curves}.py` | `potential_cache_audit.json`, `spatial_manifest.json`, `spatial_components.csv`, and every referenced complete trajectory CSV; preserve particle IDs, mass floors, cost scaling, selection rules and time grids |
| Mouse fixed-origin STE | `experiments/mean_centered_transition_20261011/rgc_origin_analysis.py` | Fixed 1,708 source IDs, original trajectory cache, spatial cache, frozen component table and selected weight version; directory naming alone does not select the current score |
| Current fish main score | `outputs/fig4_raw_initial_reference_20261018/scripts/prepare.py` | `inputs/raw_components.csv`, historical reference/HVG tables, cohort definitions (41/339) and midpoint convention; released raw-component CSV supports inspection |
| Gene gradients | `scripts/dorsal_midbrain/rank_dnb_pool_by_potential_descent.py` | Stored original 10D representations, dynamics model, recovered encoder, gene order, expression standard deviations, decoder↔dynamics affine maps, cell/time selection |
| Gene-ranking alternatives | `scripts/dorsal_midbrain/rank_absolute_potential_speed.py`, `rank_adjacent_potential_speed_change.py` | Exact `potential_descent_by_time.csv`, selected time pair, complete ranking and Table S1/figure source crosswalk |
| Mouse counterfactuals | `psg_full_ruot_gat/scripts/run_dorsal_gene_critical_perturbation.py`; `scripts/dorsal_midbrain/{make_gene_perturbation_paired_trajectory_panel,screen_literature_gene_b_to_b_transitions,make_final_perturbation_story_figures}.py` | Run summary, `stvcr_notebook_trained.h5ad`, dynamics checkpoint, `spatiotemporal_classifier.pt`, recovered GAT checkpoint and affine maps, dose/config metadata, matched control IDs |
| Zebrafish regional counterfactuals | `result_figures/斑马鱼/ij/code/make_zebrafish_ab_region_perturbation.py`; `SpaPOT_runs/zesta_scripts/make_zesta_stvcr_classifier_dynamic.py` | Selected ZESTA trained AnnData/model, `detail_mesoderm_classifier/state_classifier.pt`, fixed source branch table, spatial background/critical summary, original region and perturbation settings |
| Benchmark | `scripts/dorsal_midbrain/rebuild_figure6_unified_model_comparison_v5.py` and three included helper scripts | `v4_spatial_slice_prediction_source_manifest.csv`, `v4_main_method_manifest.csv`, completed predictions, multiseed manifest, coordinate conversion inputs and native external-method versions |

The mouse recovered GAT checkpoint exists locally at
`SpaPOT_runs/dorsal_midbrain_retrained_gat10_notebook_hybrid_e1000_s512_noalign/gene_prior_gatae_retrain_20260523/gene_prior_gatae_10d.pt`.
Its existence does not recover the original encoder: Note 6 explicitly states
that the original checkpoint is unavailable and that an affine-linked retrained
encoder is used for perturbations. Preserve that distinction in the release.
The historical `stvcr` model package and other imported helpers must also be
provided or explicitly mapped to the public model with equivalence checks.

No absent scientific algorithm is reconstructed in this patch. Sources already
found are reused; missing portable entry points, source-to-result links and data
assets remain explicit release requirements.

## Submission priority and proposed availability wording

The [official RECOMB 2027 call](https://recomb.org/recomb2027/call_for_papers)
requires public source for new software/analysis methods, sufficient README
documentation by the submission deadline, and a clear repository link in the
paper. It does not itself certify this package as compliant or require rerunning
every large experiment. The full-paper deadline is 4 November 2026 at 23:59 AoE;
abstract registration is required by 28 October at 23:59 AoE. The main paper
limit is 10 Letter pages excluding references, with the stated title-page and
appendix allowances; a journal-length DOCX is not a verified conference PDF.

Priority before submission: (1) choose and document one definition for each
reported curve/ranking; (2) retain these verified statistical entry points and
publish the corresponding source dependency bundle; (3) make weights/prepared
inputs and manifests available at stable URLs, or state precisely what cannot
be regenerated; (4) pin the audited release commit in the paper. Releasing a
source snapshot alone does not demonstrate complete raw-data reproducibility.

Suggested availability text for the coverage of this patch:

> The stCTD dynamics implementation, prepared-data training example, historical
> analysis source, and saved-score reproduction scripts for Supplementary Tables
> S4–S6 and Figure S34 are available at
> https://github.com/wanghongye200211/stCTD. The repository documents the inputs,
> analysis versions and remaining dependencies required for end-to-end
> reproduction of the manuscript results.

Use this text only with the commit/branch that actually contains the package;
do not imply that merging or a complete data release has already occurred.

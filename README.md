# CoverNET: a Class Cover Catch Digraph-inspired neural network for imbalanced classification

Code and results for the paper

> Sağlam, F. and Kara, V. *CoverNET: A Class Cover Catch Digraph-Inspired Neural Network Architecture for Imbalanced Classification.* Neural Networks (under revision, manuscript NEUNET-D-26-05591).

CoverNET replaces the linear output layer of a neural network by a **Class Cover Layer** (CCL): every class is scored by a soft union of balls with learnable dominant points and radii in the penultimate representation space. The network is trained with classification, coverage, repulsion, and radius-penalty terms and a *k*-means warm start.

## Repository layout

```
global.R                  libraries, loads everything in R/, loads data, fold creation, single-fold evaluation
R/
  factory.R               the 19 compared methods (+ matched heads, ablation variants), their hyperparameter grids
  classifiers/            one file per classifier (R6 wrappers; CoverNET and the PyTorch networks are in classifier_CCCDNNET.R)
  utils/                  preprocessing, resampling, metrics, tuner
data/
  datasets_binary.rds     the 72 binary imbalanced datasets (list of data frames, last column = class)
results/                  result tables and figures (see "Results" below)
scripts/
  first_version/          protocol of the first submission (flat cross-validation) and its plots
  revision1/              everything added in the revision (nested CV, seeds, ablation, K, timing)
apps/shiny_app.R          Shiny application for benchmarking imbalanced classifiers
```

Every script is run **from the project root** (they start with `source("global.R")` and read/write `results/...`):

```r
source("scripts/revision1/nested_cv/01_run_nested_cv.R")      # or: Rscript "scripts/revision1/nested_cv/01_run_nested_cv.R"
```

## Setup

R (4.x) and Python (3.9+) are needed. The neural networks are written in PyTorch and called from R through `reticulate`.

```r
Rscript install_packages.R          # R packages
```
```bash
pip install -r requirements.txt     # torch, numpy, scipy, scikit-learn  (point reticulate to this environment)
```
`SMOTEWB` and `bbtcomp` (Bayesian Bradley-Terry comparison, Jacques et al., 2023) must be installed if they are not available from CRAN (TODO(authors): add the install command). All experiments were run on CPU, one thread per fit.

## Data

`data/datasets_binary.rds` contains the 72 binary imbalanced datasets derived from 30 source datasets (UCI and KEEL repositories; `results/dataset info.xlsx` lists name, n, p, imbalance ratio). Datasets derived from the same source are not independent, and the family of every dataset is given in the last column of the dataset table of the paper's appendix.

TODO(authors): list the source URL / version of each dataset and how categorical features and missing values were handled.

## Reproducing the results

The experimental protocol of the revision is **nested stratified cross-validation**: 5 outer folds (test only), 3 inner folds, 20 random-search configurations per method, selection by inner AUC (305 fits per method and dataset). Seeds are fixed (`SEED <- 42`); the folds come from `make_folds()` in `global.R`.

| Step | Script (in `scripts/revision1/`) | Output (in `results/`) | Paper |
|---|---|---|---|
| Main evaluation: 19 methods + tuned RF/SVM + matched heads | `nested_cv/01_run_nested_cv.R` | `nested CV - revision 1/` | Overall Performance, Statistical Significance |
| Ranks, tests, Bradley-Terry input | `nested_cv/02_nested_cv_summary.R` | `nested CV - revision 1/`, `Bradley Terry ... nested tuned RF and SVM - revision 1.xlsx` | Overall Performance, Statistical Significance, Appendix |
| Nested vs flat | `nested_cv/03_nested_vs_flat_appendix.R` | `nested CV - revision 1/` | Appendix (nested vs flat) |
| Non-saturated datasets | `nested_cv/04_non_saturated_datasets.R` | `nested CV - revision 1/` | (response letter) |
| One unit per family | `nested_cv/05_family_sensitivity_nested.R` (nested), `06_family_sensitivity_flat.R` (flat) | `nested CV - revision 1/family sensitivity - revision 1.xlsx` | Sensitivity to Dependent Datasets |
| Matched output heads | `nested_cv/08_matched_heads_summary.R` | `nested CV - revision 1/matched heads - revision 1.xlsx` | Matched Output-Head Comparison |
| Stratified analyses (n, p, IR, p/n) | `nested_cv/07_stratified_plots.R` | `Figures/rank vs ... nested - revision 1.png` | Performance by n, p, IR, p/n |
| Ten seeds | `seeds/01_run_seeds.R`, `seeds/02_seeds_summary.R` | `seeds - revision 1/` | Stability over Random Seeds |
| Ablation | `ablation/01_run_ablation.R`, `ablation/02_ablation_family_sensitivity.R` | `ablation/` | Ablation Analysis |
| Number of dominant points | `dominant_points_K/` (`03_run_simulation.R`, `04_run_sensitivity_real_data.R`, `06_run_K_tuning_alpha.R`, `07_K_tuning_summary.R`) | `K simulation/`, `K sensitivity/`, `CoverNET K tuning - revision 1/` | Appendix (K) |
| Timing and scalability | `timing/01_run_timing.R` | `timing - revision 1.xlsx` | Computational Efficiency, Appendix |
| First version (flat CV) | `first_version/` | `results combined.xlsx`, `Bradley Terry model results.xlsx` | superseded; kept for the comparison with nested CV |

Notes:
* `01_run_nested_cv.R` is resumable (finished datasets are skipped) and is long-running; the scripts for seeds, ablation, and K use `parallel` workers (`N_CORES`).
* The raw `.rds` result objects are large and are not tracked by git (`.gitignore`); the `.xlsx`/`.csv` tables that the paper uses are included. TODO(authors): archive the `.rds` files (e.g. on Zenodo) and link them here.
* `dominant_points_K/archive_run_K_tuning_fixed_values.R` is an earlier variant (K in {2, 5, 10}) that is not used in the paper.

## Citation

TODO(authors): add the BibTeX entry and DOI once the paper is published.

## License

TODO(authors): add a license (for example MIT for the code). The datasets keep the licenses of the UCI and KEEL repositories.

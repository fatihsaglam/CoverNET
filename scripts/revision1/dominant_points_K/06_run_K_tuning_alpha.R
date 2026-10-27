# ==============================================================================
# Script: scripts/revision1/dominant_points_K/06_run_K_tuning_alpha.R   (NEW FILE, put it next to run.R; the original run.R is untouched)
# Purpose: tune the number of dominant points of CoverNET (method CCCDNNET) on top of the paper's tuning:
#          for every dataset the best configuration of the paper (architecture, lambdas; BestParams sheet of
#          results/results combined.xlsx) is re-run with alpha in ALPHA_LEVELS (factory.R), where
#          K_c = min(n_c, max(K_MIN, floor(alpha * sqrt(n_c)))); alpha = 1 is the sqrt rule. All three runs are fresh
#          (same 5 folds), the best alpha is chosen by mean ROCAUC (run_flat_cv), i.e. 3 more configurations
#          per dataset on top of the paper's 20.
# Needs: R/classifiers/classifier_CCCDNNET.R (arguments alpha, k_min) and R/factory.R (ALPHA_LEVELS, K_MIN).
# Datasets: the 72 datasets of results/dataset info.xlsx. Resumable: finished datasets are skipped on restart.
# Pilot: N_PILOT <- 15 runs 15 datasets spread over the list, written to a separate folder.
# Output: results/CoverNET K tuning - revision 1/results.rds and results.xlsx (same layout as the main results;
#         the main results folder is not touched).
# ==============================================================================

source("global.R")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

SEED         <- 42
N_FOLDS      <- 5
TUNE_METRIC  <- "ROCAUC"
EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4", "IAM", "BBrier")

N_PILOT      <- NULL                     # e.g. 15 for a pilot; NULL = all datasets
VERBOSE      <- FALSE                    # TRUE prints every config and fold
RESULTS_DIR  <- file.path("results", if (is.null(N_PILOT)) "CoverNET K tuning - revision 1"
                                     else "CoverNET K tuning pilot - revision 1")
METHOD       <- "CCCDNNET"

# best configuration of the paper per dataset (hidden_dim, n_hidden, lambda1-3)
bp <- openxlsx::read.xlsx("results/results combined.xlsx", sheet = "BestParams")
bp <- bp[bp$Method == "CoverNET" & !is.na(bp$BestParams), ]       # two datasets of the sheet have no best params (not among the 72)
get_cfg <- function(s) {
  kv <- strsplit(strsplit(s, ", ", fixed = TRUE)[[1]], "=", fixed = TRUE)
  v  <- setNames(vapply(kv, `[`, "", 2), vapply(kv, `[`, "", 1))
  data.frame(hidden_dim = as.integer(v[["hidden_dim"]]), n_hidden = as.integer(v[["n_hidden"]]),
             lambda1 = as.numeric(v[["lambda1"]]), lambda2 = as.numeric(v[["lambda2"]]),
             lambda3 = as.numeric(v[["lambda3"]]))
}
best_cfg <- setNames(lapply(bp$BestParams, get_cfg), bp$Dataset)

ds_names <- openxlsx::read.xlsx("results/dataset info.xlsx")$dataset_name
ds_names <- ds_names[ds_names %in% names(datasets_binary) & ds_names %in% names(best_cfg)]
if (!is.null(N_PILOT)) ds_names <- ds_names[unique(round(seq(1, length(ds_names), length.out = N_PILOT)))]

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------
rds_file <- file.path(RESULTS_DIR, "results.rds")
results  <- if (file.exists(rds_file)) readRDS(rds_file) else list()
todo     <- setdiff(ds_names, names(results))
n_all    <- length(ds_names)
t0       <- Sys.time()

for (j in seq_along(todo)) {
  ds   <- todo[j]
  done <- n_all - length(todo) + j - 1
  el   <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  eta  <- if (j > 1) el / (j - 1) * (length(todo) - j + 1) else NA
  cat(cyan(sprintf("\n[Dataset %d / %d | %.1f%% | elapsed %.0f min | remaining ~%s min | %s]\n",
                   done + 1, n_all, 100 * done / n_all, el,
                   if (is.na(eta)) "?" else sprintf("%.0f", eta), ds)))
  
  dataset <- datasets_binary[[ds]]
  x       <- as.matrix(dataset[, -ncol(dataset)])
  y       <- as.factor(dataset[,  ncol(dataset)])
  folds   <- make_folds(y, k = N_FOLDS, seed = SEED)
  
  grid   <- cbind(best_cfg[[ds]][rep(1, length(ALPHA_LEVELS)), ], alpha = ALPHA_LEVELS)
  method <- make_method(name = METHOD, classifier_class = classifier_CCCDNNET, tuning = TRUE,
                        fixed_params = c(nnet_fixed, list(k_min = K_MIN)), tune_grid = grid)
  res <- run_flat_cv(method, x, y, folds, verbose = VERBOSE)
  results[[ds]] <- setNames(list(res), METHOD)
  save_results(results)
}

# which alpha was chosen per dataset
chosen <- vapply(results, function(r) if (is.null(r[[METHOD]])) NA_real_ else as.numeric(r[[METHOD]]$best_params$alpha),
                 numeric(1))
cat("\nChosen alpha over the datasets:\n"); print(table(alpha = chosen, useNA = "ifany"))
cat(green("\nDone.\n"))

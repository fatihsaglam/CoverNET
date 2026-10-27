# ==============================================================================
# Script: scripts/revision1/seeds/01_run_seeds.R   (NEW FILE, put it next to run.R)
# Purpose: reviewer 3 comment 11 (single seed). The neural methods of the nested CV are re-evaluated with N_SEEDS different seeds.
#   The configuration selected by the nested CV for every outer fold (results/nested CV - revision 1/results.rds, chosen_configs)
#   is kept; only the final model is trained again with another seed and scored on the same outer test part.
#   A seed changes the weight initialisation, the minibatch order, the validation split of early stopping (numpy / torch)
#   and the resampling of SMOTE (R). Folds and selected configurations are the same in every seed.
# Methods: the neural methods that use randomness (PNN and SkewPNN are deterministic and are not repeated).
# Cost   : N_SEEDS x 5 outer folds fits per method and dataset (nested CV: 305), spread over N_CORES worker processes.
# Output : results/seeds - revision 1/results.rds  (list: dataset -> method -> matrix of seeds x metrics, mean over the 5 folds)
#          results/seeds - revision 1/results long.csv  (Dataset, Method, Seed, metrics)
#          Resumable at the level of (dataset, method).
# ==============================================================================

source("global.R")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
library(parallel)

SEED         <- 42
N_FOLDS      <- 5
TUNE_METRIC  <- "ROCAUC"
EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4", "IAM", "BBrier")

N_SEEDS      <- 10
N_DATASETS   <- NULL                       # NULL = all 72 datasets of the nested CV, or a small number for a first test
METHODS      <- c("NNET", "BalancedNNET", "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "CCCDNNET")
N_CORES      <- min(20, parallel::detectCores() - 1)
NESTED_RDS   <- file.path("results", "nested CV - revision 1", "results.rds")
OUT_DIR      <- file.path("results", "seeds - revision 1")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# task (runs on the workers): outer fold i, seed s, with the configuration selected for that outer fold
# ------------------------------------------------------------------------------
seed_task <- function(k) {
  i      <- tasks$fold[k]
  s      <- tasks$seed[k]
  method <- methods_list[[m]]
  set.seed(1000 * s + i)
  reticulate::py_run_string(sprintf("import torch, numpy as np; torch.manual_seed(%d); np.random.seed(%d)", 1000 * s + i, 1000 * s + i))
  eval_fold(method, x, y, folds[[i]], params = as.list(method$tune_grid[best[i], , drop = FALSE]), verbose = FALSE)
}

wd    <- getwd()
tasks <- expand.grid(fold = seq_len(N_FOLDS), seed = seq_len(N_SEEDS))
cl    <- makeCluster(N_CORES)
clusterExport(cl, c("wd", "SEED", "N_FOLDS", "TUNE_METRIC", "EVAL_METRICS", "tasks", "seed_task"))
invisible(clusterEvalQ(cl, {
  setwd(wd)
  source("global.R")
  Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
  reticulate::py_run_string("import torch; torch.set_num_threads(1)")
}))
clusterExport(cl, c("SEED", "N_FOLDS", "TUNE_METRIC", "EVAL_METRICS"))      # global.R overwrote them on the workers

grid_sum <- function() sapply(methods_list, function(m) if (m$tuning) sum(as.numeric(unlist(m$tune_grid))) else 0)
stopifnot(all(sapply(clusterCall(cl, grid_sum), function(s) isTRUE(all.equal(s, grid_sum())))))

# ------------------------------------------------------------------------------
# datasets and selected configurations of the nested CV
# ------------------------------------------------------------------------------
nested   <- readRDS(NESTED_RDS)
ds_names <- names(nested)
info     <- data.frame(ds = ds_names, n = sapply(ds_names, function(d) nrow(datasets_binary[[d]])))
ds_names <- ds_names[order(info$n)]
if (!is.null(N_DATASETS)) ds_names <- ds_names[round(seq(1, length(ds_names), length.out = N_DATASETS))]

rds_file <- file.path(OUT_DIR, "results.rds")
results  <- if (file.exists(rds_file)) readRDS(rds_file) else list()
n_all    <- length(ds_names) * length(METHODS)
n_done   <- sum(sapply(ds_names, function(d) sum(METHODS %in% names(results[[d]]))))
n_start  <- n_done
t0       <- Sys.time()
cat(green(sprintf("Seeds: %d datasets x %d methods x %d seeds, %d workers\n", length(ds_names), length(METHODS), N_SEEDS, N_CORES)))

for (j in seq_along(ds_names)) {
  ds      <- ds_names[j]
  dataset <- datasets_binary[[ds]]
  x       <- as.matrix(dataset[, -ncol(dataset)])
  y       <- as.factor(dataset[,  ncol(dataset)])
  folds   <- make_folds(y, k = N_FOLDS, seed = SEED)

  el  <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  eta <- if (n_done > n_start) el / (n_done - n_start) * (n_all - n_done) else NA
  cat(bold(cyan(sprintf("\n===== Dataset %d / %d | %s | n = %d, p = %d | %.1f%% | elapsed %.0f min | remaining ~%s min =====\n",
                        j, length(ds_names), ds, nrow(x), ncol(x), 100 * n_done / n_all, el,
                        if (is.na(eta)) "?" else sprintf("%.0f", eta)))))
  clusterExport(cl, c("x", "y", "folds"))

  for (m in setdiff(METHODS, names(results[[ds]]))) {
    t_m  <- Sys.time()
    best <- as.integer(strsplit(nested[[ds]][[m]]$best_params$chosen_configs, "/")[[1]])    # chosen configuration per outer fold
    stopifnot(length(best) == N_FOLDS, !anyNA(best))
    clusterExport(cl, c("m", "best"))
    sc <- do.call(rbind, parLapplyLB(cl, seq_len(nrow(tasks)), seed_task))                  # one row per (fold, seed)
    if (anyNA(sc[, TUNE_METRIC])) stop("NA in ", TUNE_METRIC, ": ", ds, " ", m)
    per_seed <- t(sapply(seq_len(N_SEEDS), function(s) colMeans(sc[tasks$seed == s, EVAL_METRICS, drop = FALSE])))
    rownames(per_seed) <- seq_len(N_SEEDS)
    results[[ds]][[m]] <- per_seed
    n_done <- n_done + 1
    saveRDS(results, rds_file)
    cat(sprintf("   %-12s %s mean %.4f, sd over seeds %.4f | %.1f min\n", m, TUNE_METRIC, mean(per_seed[, TUNE_METRIC]),
                sd(per_seed[, TUNE_METRIC]), as.numeric(difftime(Sys.time(), t_m, units = "mins"))))
  }
}
stopCluster(cl)

long <- do.call(rbind, lapply(names(results), function(d) do.call(rbind, lapply(names(results[[d]]), function(mm)
  data.frame(Dataset = d, Method = mm, Seed = seq_len(nrow(results[[d]][[mm]])), results[[d]][[mm]], row.names = NULL)))))
write.csv(long, file.path(OUT_DIR, "results long.csv"), row.names = FALSE)
cat(green("\nDone.\n"))

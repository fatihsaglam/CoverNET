# ==============================================================================
# Script: scripts/revision1/nested_cv/01_run_nested_cv.R   (NEW FILE, put it next to run.R; the original run.R is untouched)
# Purpose: nested cross-validation on a subset of the datasets (reviewer 3, comment 4).
#   Outer loop : the same 5 stratified folds as the paper (make_folds, SEED 42), used ONLY for performance estimation.
#   Inner loop : inside the training part of every outer fold, 3 stratified folds select the configuration
#                (same 20 configurations of factory.R, best mean ROCAUC). The selected configuration is refitted on the
#                whole outer training part and scored on the outer test part, which never took part in the selection.
# Parallel : the 100 tasks (5 outer folds x 20 configurations) of one method and dataset are spread over N_CORES worker
#            processes with dynamic load balancing; then the 5 refits. Datasets and methods run one after another.
# Methods  : all tuned methods of factory.R (10). Methods without tuning have no selection step, so their results of the
#            paper are unchanged and are taken from the main results.
# Datasets : N_SUBSET datasets of the 72, 4 sample size blocks, within each block spread over the imbalance ratio.
# Cost     : (20 configs x 3 inner folds + 1) x 5 outer folds = 305 fits per method and dataset (flat CV: 100).
# Output   : results/nested CV - revision 1/results.rds and results.xlsx (same layout as the main results).
#            Resumable at the level of (dataset, method).
# Memory   : every worker loads the project and all datasets; lower N_CORES if the RAM runs out.
# ==============================================================================

source("global.R")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
library(parallel)

SEED         <- 42
N_FOLDS      <- 5
K_INNER      <- 3
TUNE_METRIC  <- "ROCAUC"
EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4", "IAM", "BBrier")

N_SUBSET     <- NULL                      # NULL = all datasets of results/dataset info.xlsx, or a multiple of 4 for a subset
# METHOD_NAMES <- NULL                      # NULL = all tuned methods, or e.g. c("CCCDNNET", "NNET")
# METHOD_NAMES <- c("RFtuned", "SMOTERFtuned", "RUSRFtuned", "SVMtuned", "SMOTESVMtuned")
METHOD_NAMES <- c("MLPheadNNET", "RBFheadNNET", "CCLBCENNET")
N_CORES      <- min(20, parallel::detectCores() - 1)
RESULTS_DIR  <- file.path("results", "nested CV - revision 1")

# ------------------------------------------------------------------------------
# tasks (run on the workers; x, y, folds, m, tasks, best are exported before use)
#   inner_task(k): mean inner ROCAUC of configuration g on the training part of outer fold i
#   final_task(i): refit the selected configuration on the outer training part, score the outer test part
# ------------------------------------------------------------------------------
inner_task <- function(k) {
  i      <- tasks$fold[k]
  method <- methods_list[[m]]
  tr     <- folds[[i]]$train
  inner  <- make_folds(y[tr], k = K_INNER, seed = SEED)
  params <- as.list(method$tune_grid[tasks$g[k], , drop = FALSE])
  mean(sapply(inner, function(f)
    eval_fold(method, x[tr, , drop = FALSE], y[tr], f, params = params, verbose = FALSE)[TUNE_METRIC]),
    na.rm = TRUE)
}
final_task <- function(i) {
  method <- methods_list[[m]]
  eval_fold(method, x, y, folds[[i]], params = as.list(method$tune_grid[best[i], , drop = FALSE]), verbose = FALSE)
}

# ------------------------------------------------------------------------------
# workers: every worker loads the project (global.R) and uses one torch thread
# ------------------------------------------------------------------------------
wd    <- getwd()
tasks <- expand.grid(g = seq_len(N_RANDOM), fold = seq_len(N_FOLDS))     # g varies fastest
cl    <- makeCluster(N_CORES)
clusterExport(cl, c("wd", "SEED", "K_INNER", "TUNE_METRIC", "EVAL_METRICS", "tasks", "inner_task", "final_task"))
invisible(clusterEvalQ(cl, {
  setwd(wd)
  source("global.R")
  Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
  reticulate::py_run_string("import torch; torch.set_num_threads(1)")
}))
clusterExport(cl, c("SEED", "K_INNER", "TUNE_METRIC", "EVAL_METRICS"))   # global.R overwrote them on the workers

# the workers must hold exactly the same configuration grids as this session
grid_sum <- function() sapply(methods_list, function(m) if (m$tuning) sum(as.numeric(unlist(m$tune_grid))) else 0)
stopifnot(all(sapply(clusterCall(cl, grid_sum), function(s) isTRUE(all.equal(s, grid_sum())))))

# ------------------------------------------------------------------------------
# datasets (subset) and methods
# ------------------------------------------------------------------------------
ds72 <- openxlsx::read.xlsx("results/dataset info.xlsx")$dataset_name
info <- data.frame(ds   = names(datasets_binary),
                   n    = sapply(datasets_binary, nrow),
                   p    = sapply(datasets_binary, ncol) - 1,
                   IR   = sapply(datasets_binary, function(d) { t <- table(d[[ncol(d)]]); max(t) / min(t) }),
                   minc = sapply(datasets_binary, function(d) min(table(d[[ncol(d)]]))))
info <- info[info$ds %in% ds72, ]            # the 72 datasets of the paper (abalone_21_vs_8 has only 14 minority cases and is part of them)
info$blk <- cut(rank(info$n, ties.method = "first"), 4, labels = FALSE)
ds_names <- if (is.null(N_SUBSET)) info$ds else unlist(lapply(split(info, info$blk), function(b) {
  b <- b[order(b$IR), ]
  b$ds[unique(round(seq(1, nrow(b), length.out = N_SUBSET / 4)))]
}), use.names = FALSE)
ds_names <- ds_names[order(info$n[match(ds_names, info$ds)])]

tuned   <- names(Filter(function(m) m$tuning, methods_list))
METHODS <- if (is.null(METHOD_NAMES)) tuned else METHOD_NAMES

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------
rds_file <- file.path(RESULTS_DIR, "results.rds")
results  <- if (file.exists(rds_file)) readRDS(rds_file) else list()
n_all    <- length(ds_names) * length(METHODS)
n_done   <- sum(sapply(ds_names, function(d) sum(METHODS %in% names(results[[d]]))))
n_start  <- n_done
t0       <- Sys.time()
cat(green(sprintf("Nested CV: %d datasets x %d methods, %d workers\n", length(ds_names), length(METHODS), N_CORES)))

for (j in seq_along(ds_names)) {
  ds      <- ds_names[j]
  dataset <- datasets_binary[[ds]]
  x       <- as.matrix(dataset[, -ncol(dataset)])
  y       <- as.factor(dataset[,  ncol(dataset)])
  folds   <- make_folds(y, k = N_FOLDS, seed = SEED)
  i       <- match(ds, info$ds)

  el  <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  eta <- if (n_done > n_start) el / (n_done - n_start) * (n_all - n_done) else NA
  cat(bold(cyan(sprintf("\n===== Dataset %d / %d | %s | n = %d, p = %d, IR = %.1f | %.1f%% | elapsed %.0f min | remaining ~%s min =====\n",
                        j, length(ds_names), ds, info$n[i], info$p[i], info$IR[i], 100 * n_done / n_all, el,
                        if (is.na(eta)) "?" else sprintf("%.0f", eta)))))
  clusterExport(cl, c("x", "y", "folds"))

  for (m in setdiff(METHODS, names(results[[ds]]))) {
    t_m <- Sys.time()
    clusterExport(cl, "m")
    inner_auc <- matrix(unlist(parLapplyLB(cl, seq_len(nrow(tasks)), inner_task)), nrow = N_RANDOM)   # configs x outer folds
    inner_auc[is.nan(inner_auc)] <- -Inf
    best <- apply(inner_auc, 2, which.max)                                                            # chosen config per outer fold
    if (any(apply(inner_auc, 2, function(v) all(is.infinite(v))))) stop("all inner scores failed: ", ds, " ", m)
    clusterExport(cl, "best")
    scores <- do.call(rbind, parLapplyLB(cl, seq_len(N_FOLDS), final_task))
    if (anyNA(scores[, TUNE_METRIC])) stop("NA in outer ", TUNE_METRIC, ": ", ds, " ", m)

    results[[ds]][[m]] <- list(
      best_params = c(methods_list[[m]]$fixed_params, list(chosen_configs = paste(best, collapse = "/"))),
      cv_scores   = scores,
      mean_scores = colMeans(scores, na.rm = TRUE),
      sd_scores   = apply(scores, 2, sd, na.rm = TRUE))
    n_done <- n_done + 1
    save_results(results)
    cat(sprintf("   %-12s %s %.4f | %.1f min\n", m, TUNE_METRIC, results[[ds]][[m]]$mean_scores[TUNE_METRIC],
                as.numeric(difftime(Sys.time(), t_m, units = "mins"))))
  }
}
stopCluster(cl)
cat(green("\nDone.\n"))

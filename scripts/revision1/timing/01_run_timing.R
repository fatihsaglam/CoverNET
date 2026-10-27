# ==============================================================================
# Script: scripts/revision1/timing/01_run_timing.R   (NEW FILE, put it next to run.R)
# Purpose: reviewer 3 comment 10 (total computational cost), reviewer 1 comment 7 and reviewer 2 comment 4 (scalability).
#   Part A  final fit time of every method on the 72 datasets, taken from the nested CV results (no new run).
#   Part B  hyperparameter search time and k-means time on N_DS datasets spread over the sample size: every one of the 20
#           configurations of each method is fitted once (outer fold 1 training part) and timed. The k-means initialisation of
#           CoverNET is timed separately for every configuration.
#   Part C  scalability on synthetic Gaussian data (imbalance ratio 20): fit time, k-means time and the size of the largest
#           tensor of the CCL (batch x K x d) of NN and CoverNET for growing n and p.
# All times are CPU, one thread, seconds. Parts B and C run one after another (no parallel workers), so timings are comparable.
# Output : results/timing - revision 1.xlsx   (sheets: Final fit time, Search time, Scaling)
# ==============================================================================

source("global.R")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
reticulate::py_run_string("import torch; torch.set_num_threads(1)")

SEED        <- 42
N_FOLDS     <- 5
EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4", "IAM", "BBrier")
N_DS        <- 12                                    # datasets of part B
METHODS_B   <- c("NNET", "SMOTENNET", "FocalNNET", "CCCDNNET")
SCALE_N     <- c(1e3, 1e4, 1e5)                      # part C
SCALE_P     <- c(10, 100, 500)
OUT_FILE    <- "results/timing - revision 1.xlsx"
NESTED      <- "results/nested CV - revision 1/results.xlsx"

# ------------------------------------------------------------------------------
# Part A: final fit time from the nested CV (fit + predict of the final model, mean over the 5 outer folds)
# ------------------------------------------------------------------------------
nested <- openxlsx::read.xlsx(NESTED, sheet = "Means")
rt     <- nested[nested$Metric == "Runtime", ]
rt$Dataset <- NULL; rt$Metric <- NULL
neural <- c("NNET", "BalancedNNET", "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "PNN", "SkewPNN", "CCCDNNET")
part_a <- data.frame(Method = names(rt), Median = round(sapply(rt, median), 2),
                     Q1 = round(sapply(rt, quantile, 0.25), 2), Q3 = round(sapply(rt, quantile, 0.75), 2),
                     Max = round(sapply(rt, max), 2),
                     MedianRatioToNN = round(sapply(rt, function(v) median(v / rt$NNET)), 2), row.names = NULL)
cat("Part A, final fit time in seconds (median over the 72 datasets):\n"); print(part_a)

# ------------------------------------------------------------------------------
# k-means time of the CoverNET initialisation (same steps as train_cccd: hidden representation, then k-means per class)
# ------------------------------------------------------------------------------
py_run_string("
import time, numpy as np, torch
from sklearn.cluster import KMeans
def time_kmeans(x_train, y_train, p, h, n_layers, k_pos, k_neg):
    x = np.array(x_train, dtype=np.float32); y = np.array(y_train, dtype=np.int32).flatten()
    model = NetCCCD(int(p), int(h), int(n_layers), 'relu', int(k_pos), int(k_neg))
    t0 = time.time()
    with torch.no_grad():
        hh = model.hidden(torch.tensor(x)).numpy()
    for hp, k in ((hh[y == 1], int(k_pos)), (hh[y == 0], int(k_neg))):
        k = max(1, min(k, len(hp)))
        if k > 1:
            KMeans(n_clusters=k, n_init=10, random_state=42).fit(hp)
    return time.time() - t0
")

# ------------------------------------------------------------------------------
# Part B: search time. Every configuration once, on the training part of outer fold 1.
# ------------------------------------------------------------------------------
ds72  <- openxlsx::read.xlsx("results/dataset info.xlsx")
ds72  <- ds72[order(ds72$n), ]
pick  <- ds72$dataset_name[unique(round(seq(1, nrow(ds72), length.out = N_DS)))]
part_b <- list()
cat(green(sprintf("\nPart B: %d datasets x %d methods x 20 configurations\n", length(pick), length(METHODS_B))))
for (j in seq_along(pick)) {
  ds <- pick[j]; d <- datasets_binary[[ds]]
  x <- as.matrix(d[, -ncol(d)]); y <- as.factor(d[, ncol(d)])
  folds <- make_folds(y, k = N_FOLDS, seed = SEED)
  cat(bold(cyan(sprintf("\n===== Dataset %d / %d | %s | n = %d, p = %d =====\n", j, length(pick), ds, nrow(x), ncol(x)))))
  for (m in METHODS_B) {
    method <- methods_list[[m]]
    t_fit  <- numeric(nrow(method$tune_grid)); t_km <- numeric(nrow(method$tune_grid))
    for (g in seq_len(nrow(method$tune_grid))) {
      params   <- as.list(method$tune_grid[g, , drop = FALSE])
      t_fit[g] <- eval_fold(method, x, y, folds[[1]], params = params, verbose = FALSE)["Runtime"]
      if (m == "CCCDNNET") {
        tr   <- folds[[1]]$train
        xz   <- scale(x[tr, , drop = FALSE]); xz[is.na(xz)] <- 0; xz <- matrix(xz, nrow(xz))
        yb   <- as.numeric(y[tr] == names(which.min(table(y[tr]))))
        t_km[g] <- py$time_kmeans(xz, yb, ncol(x), params$hidden_dim, params$n_hidden,
                                  max(1L, floor(sqrt(sum(yb == 1)))), max(1L, floor(sqrt(sum(yb == 0)))))
      }
    }
    part_b[[length(part_b) + 1]] <- data.frame(
      Dataset = ds, n = nrow(x), p = ncol(x), Method = m,
      SearchOneFold = round(sum(t_fit, na.rm = TRUE), 1),                         # 20 configurations, one fold
      SearchFlatCV  = round(5 * sum(t_fit, na.rm = TRUE), 1),                     # 20 configurations x 5 folds (flat protocol)
      MedianFit     = round(median(t_fit, na.rm = TRUE), 2),
      KmeansPerFit  = round(median(t_km), 2), KmeansShare = round(sum(t_km) / sum(t_fit, na.rm = TRUE), 3))
    cat(sprintf("   %-10s search (20 configs, 1 fold) %.1f s | median fit %.2f s%s\n", m, sum(t_fit, na.rm = TRUE),
                median(t_fit, na.rm = TRUE), if (m == "CCCDNNET") sprintf(" | k-means share %.0f%%", 100 * sum(t_km) / sum(t_fit, na.rm = TRUE)) else ""))
  }
}
part_b <- do.call(rbind, part_b)

# ------------------------------------------------------------------------------
# Part C: scalability on synthetic data (two Gaussian classes, imbalance ratio 20, hidden 64 x 2 layers)
# ------------------------------------------------------------------------------
part_c <- list()
cat(green("\nPart C: synthetic scalability\n"))
for (n in SCALE_N) for (p in SCALE_P) {
  set.seed(SEED)
  n_pos <- round(n / 21); n_neg <- n - n_pos
  shift <- rnorm(p); shift <- 1.5 * shift / sqrt(sum(shift^2))
  x <- rbind(matrix(rnorm(n_neg * p), n_neg, p), matrix(rnorm(n_pos * p), n_pos, p) + rep(shift, each = n_pos))
  y <- factor(c(rep("neg", n_neg), rep("pos", n_pos)), levels = c("neg", "pos"))
  K <- c(max(1L, floor(sqrt(n_pos))), max(1L, floor(sqrt(n_neg))))
  for (m in c("NNET", "CCCDNNET")) {
    clf <- methods_list[[m]]$classifier_class$new()
    t0  <- proc.time()["elapsed"]
    do.call(clf$fit, c(list(x = x, y = y), c(methods_list[[m]]$fixed_params, list(hidden_dim = 64L, n_hidden = 2L))))
    t_fit <- as.numeric(proc.time()["elapsed"] - t0)
    t_km  <- if (m == "CCCDNNET") py$time_kmeans(x, as.numeric(y == "pos"), p, 64L, 2L, K[1], K[2]) else NA
    part_c[[length(part_c) + 1]] <- data.frame(
      Method = m, n = n, p = p, KPos = if (m == "CCCDNNET") K[1] else NA, KNeg = if (m == "CCCDNNET") K[2] else NA,
      FitSeconds = round(t_fit, 1), KmeansSeconds = round(t_km, 1),
      CCLTensorMB = if (m == "CCCDNNET") round(256 * sum(K) * 64 * 4 / 2^20, 1) else NA)
    cat(sprintf("   %-9s n = %-7d p = %-4d fit %.1f s%s\n", m, n, p, t_fit, if (m == "CCCDNNET") sprintf(" (k-means %.1f s)", t_km) else ""))
    py_run_string("flush_memory()")
  }
}
part_c <- do.call(rbind, part_c)

openxlsx::write.xlsx(list("Final fit time" = part_a, "Search time" = part_b, "Scaling" = part_c), OUT_FILE)
cat(green(sprintf("\nDone. Written to %s\n", OUT_FILE)))

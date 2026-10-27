# ==============================================================================
# Script: scripts/revision1/dominant_points_K/04_run_sensitivity_real_data.R   (NEW FILE, put it next to run.R)
# Purpose: How sensitive is CoverNET to the number of dominant points K_c? (heuristic K_c = floor(sqrt(n_c)))
#
# Rules compared (everything else = the dataset's best flat-CV configuration from the paper, 72 real datasets):
#   sqrt  : K_c = max(1, floor(sqrt(n_c)))       per class        -> CoverNet (the paper's rule)
#   equal : K_c = max(1, floor(sqrt(n_minority))) for BOTH classes -> same K for both classes
#   log   : K_c = max(1, floor(log2(n_c)))       per class
#   k1, k5, k10, k20 : the same fixed K for both classes (K = 1, 5, 10, 20)
# n_c is the class size in the training part of the fold; K_c is capped at n_c (a class cannot have more
# dominant points than training cases). The K actually used is stored in the raw results (Kpos, Kneg).
#
# Same 5 stratified folds (make_folds, SEED = 42), same z-score preprocessing as eval_fold(); every rule uses the
# same torch/numpy seed per run (same hidden init, validation split and batch order) -> paired comparison.
# Reported (as in the paper and the ablation): per-dataset ranks, win/tie/loss, Friedman and Holm-adjusted Wilcoxon;
# raw metric values are never averaged across datasets.
#
# Needs: R/classifiers/classifier_CCCDNNET_ablation.R (same file as for the ablation).
# Output (results/K sensitivity/): k sensitivity raw - revision 1.rds|csv, k sensitivity summary - revision 1.csv,
#         k sensitivity by minority size - revision 1.csv, k sensitivity cost - revision 1.csv
# Cost  : 72 datasets x 5 folds x 7 rules x 3 seeds = 7560 fits (like the ablation). Start with SEEDS <- 1.
# ==============================================================================

source("global.R")

FORCE_CPU <- TRUE
if (FORCE_CPU) Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4", "IAM", "BBrier")

# ------------------------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------------------------
SEEDS   <- 1:3                              # use 1 for a first dry run
OUT_DIR <- file.path("results", "K sensitivity")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

K_RULES <- list(                            # function(n_minority, n_majority) -> c(K_minority, K_majority)
  sqrt  = function(n1, n0) c(floor(sqrt(n1)), floor(sqrt(n0))),
  equal = function(n1, n0) rep(floor(sqrt(n1)), 2),
  log   = function(n1, n0) c(floor(log2(n1)), floor(log2(n0))),
  k1    = function(n1, n0) c(1, 1),
  k5    = function(n1, n0) c(5, 5),
  k10   = function(n1, n0) c(10, 10),
  k20   = function(n1, n0) c(20, 20)
)
rule_label <- c(sqrt = "CoverNet", equal = "Equal K", log = "Log K",
                k1 = "K = 1", k5 = "K = 5", k10 = "K = 10", k20 = "K = 20")
REFERENCE <- "sqrt"

ds_names <- openxlsx::read.xlsx("results/dataset info.xlsx")$dataset_name
ds_names <- ds_names[ds_names %in% names(datasets_binary)]
# ds_names <- ds_names[1:2]                 # <- uncomment for a quick smoke test

# ------------------------------------------------------------------------------
# 2. BEST CoverNET CONFIGURATION PER DATASET (parsed from the BestParams sheet)
# ------------------------------------------------------------------------------
parse_params <- function(s) {
  kv  <- strsplit(strsplit(s, ", ", fixed = TRUE)[[1]], "=", fixed = TRUE)
  out <- lapply(kv, function(z) {
    num <- suppressWarnings(as.numeric(z[2]))
    if (is.na(num)) z[2] else num
  })
  names(out) <- vapply(kv, function(z) z[1], character(1))
  for (k in intersect(c("hidden_dim", "n_hidden", "epochs", "batch_size", "patience"), names(out)))
    out[[k]] <- as.integer(out[[k]])
  out
}

bp <- openxlsx::read.xlsx("results/results combined.xlsx", sheet = "BestParams")
bp <- bp[bp$Method == "CoverNET" & bp$Dataset %in% ds_names, ]
stopifnot(!anyDuplicated(bp$Dataset))
best_params <- setNames(lapply(bp$BestParams, parse_params), bp$Dataset)
ds_names <- intersect(ds_names, names(best_params))
cat(green(sprintf("K sensitivity: %d datasets x %d rules x %d seeds x %d folds\n",
                  length(ds_names), length(K_RULES), length(SEEDS), N_FOLDS)))

# ------------------------------------------------------------------------------
# 3. ONE FOLD
# ------------------------------------------------------------------------------
eval_fold_k <- function(params, rule, x, y, fold) {
  x_train <- x[fold$train, , drop = FALSE]
  y_train <- y[fold$train]
  x_test  <- x[fold$test,  , drop = FALSE]
  y_test  <- y[fold$test]

  # class sizes in the training part; the minority class is the positive class of the classifier
  tab <- table(y_train)
  n1  <- as.numeric(min(tab)); n0 <- as.numeric(max(tab))
  K   <- pmax(1, floor(K_RULES[[rule]](n1, n0)))
  K   <- pmin(K, c(n1, n0))                              # cannot exceed the class size
  params$k_pos <- as.integer(K[1])                       # minority class
  params$k_neg <- as.integer(K[2])                       # majority class

  prep    <- DataPreprocessor$new()
  x_train <- prep$fit_transform(x_train, y_train, scale_method = "z-score")
  x_test  <- prep$transform(x_test)

  clf     <- classifier_CCCDNNET_abl$new()
  t_start <- proc.time()["elapsed"]
  do.call(clf$fit, c(list(x = x_train, y = y_train), params))
  res     <- clf$predict(x_test)
  runtime <- as.numeric(proc.time()["elapsed"] - t_start)

  scores <- sapply(EVAL_METRICS, function(m) {
    tryCatch(metricCls_all[[m]](truth = y_test, pred = res$pred, prob = res$prob),
             error = function(e) NA_real_)
  })

  py_run_string("flush_memory()")
  c(scores, Runtime = runtime, Epochs = clf$n_epochs, BestEpoch = clf$best_epoch,
    InitSec = clf$init_seconds, TrainSec = clf$train_seconds,
    Kpos = params$k_pos, Kneg = params$k_neg)
}

# ------------------------------------------------------------------------------
# 4. MAIN LOOP (resumable: finished datasets are skipped on restart)
# ------------------------------------------------------------------------------
raw_file <- file.path(OUT_DIR, "k sensitivity raw - revision 1.rds")
raw      <- if (file.exists(raw_file)) readRDS(raw_file) else NULL
done_ds  <- if (is.null(raw)) character(0) else unique(raw$Dataset)

for (ds in setdiff(ds_names, done_ds)) {
  cat(cyan(sprintf("\n[Dataset: %s]\n", ds)))

  dataset <- datasets_binary[[ds]]
  x       <- as.matrix(dataset[, -ncol(dataset)])
  y       <- as.factor(dataset[,  ncol(dataset)])
  folds   <- make_folds(y, k = N_FOLDS, seed = SEED)
  base    <- best_params[[ds]]
  rows    <- list()

  for (sd_i in SEEDS) {
    for (r in names(K_RULES)) {
      params      <- base
      params$seed <- sd_i
      for (i in seq_along(folds)) {
        sc <- tryCatch(
          eval_fold_k(params, r, x, y, folds[[i]]),
          error = function(e) {
            message(red(sprintf("  FAILED %s | seed %d | %s | fold %d: %s",
                                ds, sd_i, r, i, conditionMessage(e))))
            NULL
          })
        if (is.null(sc)) next
        rows[[length(rows) + 1]] <- data.frame(
          Dataset = ds, Rule = r, Seed = sd_i, Fold = i, t(sc),
          check.names = FALSE, stringsAsFactors = FALSE)
      }
      cat(silver(sprintf("  seed %d | %-6s done\n", sd_i, r)))
    }
  }

  raw <- rbind(raw, do.call(rbind, rows))
  saveRDS(raw, raw_file)
  write.csv(transform(raw, Rule = rule_label[Rule]),
            file.path(OUT_DIR, "k sensitivity raw - revision 1.csv"), row.names = FALSE)
}

# ------------------------------------------------------------------------------
# 5. SUMMARY: mean over folds and seeds -> one value per dataset x rule; then ranks across rules,
#    Friedman, paired Wilcoxon (Holm) of every rule against the reference (CoverNet). No averaging of
#    raw metric values across datasets.
# ------------------------------------------------------------------------------
METRICS_REPORT <- c("MCC", "P4", "IAM", "ROCAUC")
rule_order     <- names(K_RULES)

# minority class size of every dataset (for the breakdown below)
info <- openxlsx::read.xlsx("results/dataset info.xlsx")
n_min <- setNames(vapply(strsplit(as.character(info$Class_dist), "/", fixed = TRUE),
                         function(z) min(as.numeric(z)), numeric(1)), info$dataset_name)

summary_rows <- list(); size_rows <- list()
for (m in METRICS_REPORT) {
  per_ds <- aggregate(raw[[m]], by = list(Dataset = raw$Dataset, Rule = raw$Rule), FUN = mean, na.rm = TRUE)
  mat <- tapply(per_ds$x, list(per_ds$Dataset, per_ds$Rule), mean)[, rule_order, drop = FALSE]
  mat <- mat[complete.cases(mat), , drop = FALSE]

  ranks     <- t(apply(-mat, 1, rank))                 # rank 1 = best (higher metric is better)
  mean_rank <- colMeans(ranks)
  p_fried   <- tryCatch(friedman.test(mat)$p.value, error = function(e) NA_real_)

  others <- setdiff(rule_order, REFERENCE)
  p_raw  <- sapply(others, function(v)
    tryCatch(wilcox.test(mat[, REFERENCE], mat[, v], paired = TRUE, exact = FALSE)$p.value,
             error = function(e) NA_real_))
  p_holm <- p.adjust(p_raw, method = "holm")

  for (v in rule_order) {
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      Metric = m, Rule = unname(rule_label[v]), `N datasets` = nrow(mat),
      `Mean rank` = unname(mean_rank[v]),
      `CoverNet wins`   = if (v == REFERENCE) NA else sum(mat[, REFERENCE] >  mat[, v]),
      `Ties`            = if (v == REFERENCE) NA else sum(mat[, REFERENCE] == mat[, v]),
      `CoverNet losses` = if (v == REFERENCE) NA else sum(mat[, REFERENCE] <  mat[, v]),
      `Holm adjusted p (Wilcoxon)` = if (v == REFERENCE) NA else unname(p_holm[v]),
      `Friedman p` = p_fried,
      stringsAsFactors = FALSE, check.names = FALSE)
  }

  # mean rank of every rule within tertiles of the minority class size (equal-sized thirds of the datasets)
  nm       <- n_min[rownames(mat)]
  size_grp <- factor(ceiling(3 * rank(nm, ties.method = "first") / length(nm)), levels = 1:3,
                     labels = c("small minority", "medium minority", "large minority"))
  for (g in levels(size_grp)) {
    ii  <- size_grp == g
    mr  <- sapply(rule_order, function(v) mean(ranks[ii, v]))
    names(mr) <- unname(rule_label[rule_order])
    size_rows[[length(size_rows) + 1]] <- cbind(
      data.frame(Metric = m, `Minority size group` = g, `N datasets` = sum(ii), check.names = FALSE,
                 stringsAsFactors = FALSE),
      data.frame(as.list(mr), check.names = FALSE))
  }
}
summary_df <- do.call(rbind, summary_rows)
size_df    <- do.call(rbind, size_rows)
write.csv(summary_df, file.path(OUT_DIR, "k sensitivity summary - revision 1.csv"), row.names = FALSE)
write.csv(size_df,    file.path(OUT_DIR, "k sensitivity by minority size - revision 1.csv"), row.names = FALSE)
print(summary_df, digits = 3)
print(size_df, digits = 3)

# cost: epochs and time per rule, plus the K that was actually used (average over folds)
cost_df <- aggregate(raw[, c("Kpos", "Kneg", "Epochs", "BestEpoch", "InitSec", "TrainSec", "Runtime")],
                     by = list(Rule = raw$Rule), FUN = mean, na.rm = TRUE)
cost_df$Rule <- unname(rule_label[cost_df$Rule])
write.csv(cost_df, file.path(OUT_DIR, "k sensitivity cost - revision 1.csv"), row.names = FALSE)
print(cost_df, digits = 3)

cat(green("\nDone.\n"))

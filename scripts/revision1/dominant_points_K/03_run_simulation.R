# ==============================================================================
# Script: scripts/revision1/dominant_points_K/03_run_simulation.R   (NEW FILE, put it next to run.R)
# Purpose: Which rule for the number of dominant points K_c is the most sensible? Controlled simulation,
#          where sample size, dimension, imbalance ratio and class structure are known (Reviewer 1 / Reviewer 3, K_c choice).
#
# Data: two-class Gaussian mixtures with identity covariance. Every class is a mixture of m equally weighted
#   components (m = 1: one blob per class; m = 3: three sub-clusters per class), so K_c has something to cover.
#   The geometry (the two classes shifted by +-0.5 along a random direction, plus sub-cluster offsets of norm about
#   COMP_REL) is drawn once per dataset; the whole constellation is then multiplied by a SCALE that is calibrated by
#   bisection so that the Bayes-optimal classifier (true densities) reaches AUC = TARGET_AUC (moderate overlap).
#   Scale -> 0 gives AUC 0.5 and scale -> infinity gives AUC 1, so the target is always reachable. Overlap is therefore
#   the same in every cell, and only n, p, IR and m change.
#   (Version 1 drew the sub-cluster offsets independently for the two classes; for p = 20, m = 3 the sub-clusters were
#   so far apart that AUC was 1 whatever the class shift, so the target was not reachable. Those results are void.)
# Design (defaults): n in {500, 2000, 5000} x p in {5, 20} x IR in {3, 10, 30} x m in {1, 3} = 36 cells,
#   N_REP independent datasets per cell. The minority class has n / (1 + IR) cases.
#
# Rules (same as in the real-data study, "scripts/revision1/dominant_points_K/04_run_sensitivity_real_data.R"):
#   sqrt  : K_c = floor(sqrt(n_c)) per class (CoverNet)        equal : floor(sqrt(n_minority)) for both classes
#   log   : K_c = floor(log2(n_c)) per class                   k1, k5, k10, k20 : the same fixed K for both classes
#   s05, s025, s01 : K_c = max(1, floor(a * sqrt(n_c))) with a = 0.5, 0.25, 0.1 (a = 1 is the sqrt rule = CoverNet)
# n_c = class size in the training part of the fold; K_c is capped at n_c; the K actually used is stored (Kpos, Kneg).
#
# Network settings are the SAME fixed values for every dataset and rule (no tuning), so no rule is favoured by
# hyper-parameters that were tuned under it (a difference to the real-data study). Version 1 used epochs = 100 and
# patience = 10; about half of all fits (90% of CoverNet-rule fits at n = 5000) stopped at the 100-epoch limit, i.e.
# they were still improving. Now EPOCHS = 100, PATIENCE = 10 again, to match nnet fixed of the real-data study (a pilot showed 300 / 20 did not help CoverNet). Same 5 stratified folds
# (make_folds, SEED = 42) and z-score preprocessing as eval_fold(); every rule uses the same torch/numpy seed per
# run, so rules are compared paired.
#
# Reported: per-dataset ranks across the 7 rules, mean ranks overall / by n / p / IR / m / cell, win-tie-loss,
# Friedman and Holm-adjusted paired Wilcoxon of every rule against CoverNet. Raw metric values are never averaged
# across datasets. Independent simulated datasets => no dependence between datasets (unlike benchmark families).
#
# Needs: R/classifiers/classifier_CCCDNNET_ablation.R (same file as for the ablation).
# Lambdas: set LAMBDA in the settings; every lambda set writes to its own subfolder results/K simulation/lambda <l1> <l2> <l3>/.
# Output (in that subfolder): k simulation raw - revision 1.rds|csv, k simulation summary - revision 1.csv,
#   k simulation by factor - revision 1.csv, k simulation by cell - revision 1.csv, k simulation cost - revision 1.csv
# Cost  : 36 cells x N_REP x 5 folds x 10 rules x length(SEEDS) fits = 18000 with N_REP = 10 and one seed.
#         Dry run: N_REP <- 1 and keep only a few cells (see the smoke-test line below).
# ==============================================================================

source("global.R")

FORCE_CPU <- TRUE
if (FORCE_CPU) Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4", "IAM", "BBrier")

# ------------------------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------------------------
SEEDS      <- 1                              # network seeds per dataset (replicate datasets already carry the variance)
N_REP      <- 10                             # independent datasets per cell; use 1 for a dry run
TARGET_AUC <- 0.85                           # Bayes AUC = moderate overlap (0.75 = heavy, 0.95 = light)
COMP_REL   <- 1.5                            # norm of the sub-cluster offsets relative to the class shift (1.0); used when m > 1
EPOCHS     <- 100L                           # same as nnet fixed of the real-data study (relu, 100, 256, 1e-3, 10)
PATIENCE   <- 10L                            # early-stopping patience (real-data study: 10)
LAMBDA     <- c(lambda1 = 0.1, lambda2 = 0.1, lambda3 = 0.01)   # loss weights (cover, repel, radius penalty); try 0.01 too
OUT_DIR    <- file.path("results", "K simulation", sprintf("lambda %s", paste(LAMBDA, collapse = " ")))   # one folder per lambda set
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

GRID <- expand.grid(n = c(500, 2000, 5000), p = c(5, 20), IR = c(3, 10, 30), Clusters = c(1, 3))
GRID$Cell <- sprintf("n%d p%d IR%d m%d", GRID$n, GRID$p, GRID$IR, GRID$Clusters)
# GRID <- GRID[GRID$n == 500 & GRID$IR == 10, ]    # <- uncomment for a quick smoke test (4 cells)

K_RULES <- list(
  sqrt  = function(n1, n0) c(floor(sqrt(n1)), floor(sqrt(n0))),
  equal = function(n1, n0) rep(floor(sqrt(n1)), 2),
  log   = function(n1, n0) c(floor(log2(n1)), floor(log2(n0))),
  k1    = function(n1, n0) c(1, 1),
  k5    = function(n1, n0) c(5, 5),
  k10   = function(n1, n0) c(10, 10),
  k20   = function(n1, n0) c(20, 20),
  s05   = function(n1, n0) floor(0.5  * sqrt(c(n1, n0))),
  s025  = function(n1, n0) floor(0.25 * sqrt(c(n1, n0))),
  s01   = function(n1, n0) floor(0.1  * sqrt(c(n1, n0)))
)
rule_label <- c(sqrt = "CoverNet", equal = "Equal K", log = "Log K",
                k1 = "K = 1", k5 = "K = 5", k10 = "K = 10", k20 = "K = 20",
                s05 = "a = 0.5", s025 = "a = 0.25", s01 = "a = 0.1")
REFERENCE <- "sqrt"

# fixed network settings for every dataset and rule (class defaults, no tuning)
NET_PARAMS <- list(hidden_dim = 32L, n_hidden = 2L, activation = "relu", epochs = EPOCHS,
                   batch_size = 256L, lr = 1e-3, lambda1 = LAMBDA[["lambda1"]], lambda2 = LAMBDA[["lambda2"]],
                   lambda3 = LAMBDA[["lambda3"]], patience = PATIENCE)

# ------------------------------------------------------------------------------
# 2. DATA GENERATOR (with calibration of the class distance to the target Bayes AUC)
# ------------------------------------------------------------------------------
# log density (up to a constant) of an equal-weight mixture of N(centre_k, I) at the rows of x
mix_logdens <- function(x, centres) {
  ld <- apply(centres, 1, function(mu) -0.5 * rowSums(sweep(x, 2, mu)^2))   # N x m
  ld <- matrix(ld, nrow = nrow(x))
  mx <- apply(ld, 1, max)
  mx + log(rowMeans(exp(ld - mx)))
}

auc_rank <- function(score, is_pos) {            # Mann-Whitney AUC
  r  <- rank(score)
  n1 <- sum(is_pos); n0 <- sum(!is_pos)
  (sum(r[is_pos]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# centres of the components of both classes for a given scale (class shift +-0.5 * u plus offsets, all times scale)
make_centres <- function(u, off1, off0, scale) {
  list(c1 = scale * sweep(off1, 2,  0.5 * u, "+"),
       c0 = scale * sweep(off0, 2, -0.5 * u, "+"))
}

draw_class <- function(N, centres) {
  m    <- nrow(centres)
  comp <- sample.int(m, N, replace = TRUE)
  centres[comp, , drop = FALSE] + matrix(rnorm(N * ncol(centres)), N, ncol(centres))
}

bayes_auc <- function(u, off1, off0, scale, seed, N = 3000) {
  set.seed(seed)                                  # same noise for every scale -> smooth in scale
  cc <- make_centres(u, off1, off0, scale)
  x1 <- draw_class(N, cc$c1); x0 <- draw_class(N, cc$c0)
  x  <- rbind(x1, x0)
  sc <- mix_logdens(x, cc$c1) - mix_logdens(x, cc$c0)
  auc_rank(sc, c(rep(TRUE, N), rep(FALSE, N)))
}

make_sim_data <- function(n, p, IR, m, seed) {
  set.seed(seed)
  n1 <- max(10L, round(n / (1 + IR))); n0 <- n - n1
  u    <- rnorm(p); u <- u / sqrt(sum(u^2))
  sd_off <- if (m > 1) COMP_REL / sqrt(p) else 0  # offset norm is about COMP_REL whatever p is
  off1 <- matrix(rnorm(m * p, sd = sd_off), m, p)
  off0 <- matrix(rnorm(m * p, sd = sd_off), m, p)
  
  lo <- 0; hi <- 30                               # bisection on the scale of the whole constellation
  for (it in 1:16) {
    mid <- (lo + hi) / 2
    if (bayes_auc(u, off1, off0, mid, seed + 1) < TARGET_AUC) lo <- mid else hi <- mid
  }
  delta <- (lo + hi) / 2                          # = calibrated scale
  cc    <- make_centres(u, off1, off0, delta)
  
  set.seed(seed + 2)
  x <- rbind(draw_class(n1, cc$c1), draw_class(n0, cc$c0))
  y <- factor(c(rep("minority", n1), rep("majority", n0)), levels = c("majority", "minority"))
  # Bayes AUC of this very sample (reported for transparency)
  sc <- mix_logdens(x, cc$c1) - mix_logdens(x, cc$c0)
  list(x = x, y = y, bayes_auc = auc_rank(sc, y == "minority"), delta = delta)
}

# ------------------------------------------------------------------------------
# 3. ONE FOLD
# ------------------------------------------------------------------------------
eval_fold_k <- function(params, rule, x, y, fold) {
  x_train <- x[fold$train, , drop = FALSE]
  y_train <- y[fold$train]
  x_test  <- x[fold$test,  , drop = FALSE]
  y_test  <- y[fold$test]
  
  tab <- table(y_train)
  n1  <- as.numeric(min(tab)); n0 <- as.numeric(max(tab))
  K   <- pmax(1, floor(K_RULES[[rule]](n1, n0)))
  K   <- pmin(K, c(n1, n0))
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
raw_file <- file.path(OUT_DIR, "k simulation raw - revision 1.rds")
raw      <- if (file.exists(raw_file)) readRDS(raw_file) else NULL
done_ds  <- if (is.null(raw)) character(0) else unique(raw$Dataset)

cat(green(sprintf("K simulation: %d cells x %d datasets x %d rules x %d seeds x %d folds\n",
                  nrow(GRID), N_REP, length(K_RULES), length(SEEDS), N_FOLDS)))

# datasets that still miss at least one rule (rules added later are computed on the finished datasets, same data)
all_ds    <- unlist(lapply(seq_len(nrow(GRID)), function(g) sprintf("%s rep%02d", GRID$Cell[g], seq_len(N_REP))))
n_total   <- sum(vapply(all_ds, function(d) length(setdiff(names(K_RULES), unique(raw$Rule[raw$Dataset == d]))) > 0, logical(1)))
n_done    <- 0
n_start   <- 0
t_start_all <- Sys.time()

for (g in seq_len(nrow(GRID))) {
  for (rep_i in seq_len(N_REP)) {
    ds <- sprintf("%s rep%02d", GRID$Cell[g], rep_i)
    rules_todo <- setdiff(names(K_RULES), unique(raw$Rule[raw$Dataset == ds]))
    if (!length(rules_todo)) next
    el_min  <- as.numeric(difftime(Sys.time(), t_start_all, units = "mins"))
    eta_min <- if (n_done > n_start) el_min / (n_done - n_start) * (n_total - n_done) else NA
    cat(cyan(sprintf("\n[Dataset %d / %d | %.1f%% | elapsed %.0f min | remaining ~%s min | %s]\n",
                     n_done + 1, n_total, 100 * n_done / n_total, el_min,
                     if (is.na(eta_min)) "?" else sprintf("%.0f", eta_min), ds)))
    
    data_seed <- 100000 + g * 1000 + rep_i * 10          # fixed -> the same simulated data on every run
    sim   <- make_sim_data(GRID$n[g], GRID$p[g], GRID$IR[g], GRID$Clusters[g], data_seed)
    cat(silver(sprintf("  n1 = %d, n0 = %d, Bayes AUC = %.3f\n",
                       sum(sim$y == "minority"), sum(sim$y == "majority"), sim$bayes_auc)))
    if (abs(sim$bayes_auc - TARGET_AUC) > 0.04)
      message(red(sprintf("  WARNING: Bayes AUC %.3f is far from the target %.2f (small minority sample?)",
                          sim$bayes_auc, TARGET_AUC)))
    folds <- make_folds(sim$y, k = N_FOLDS, seed = SEED)
    rows  <- list()
    
    for (sd_i in SEEDS) {
      for (r in rules_todo) {
        params      <- NET_PARAMS
        params$seed <- sd_i
        for (i in seq_along(folds)) {
          sc <- tryCatch(
            eval_fold_k(params, r, sim$x, sim$y, folds[[i]]),
            error = function(e) {
              message(red(sprintf("  FAILED %s | seed %d | %s | fold %d: %s",
                                  ds, sd_i, r, i, conditionMessage(e))))
              NULL
            })
          if (is.null(sc)) next
          rows[[length(rows) + 1]] <- data.frame(
            Dataset = ds, Cell = GRID$Cell[g], n = GRID$n[g], p = GRID$p[g], IR = GRID$IR[g],
            `Clusters per class` = GRID$Clusters[g], Rep = rep_i, `Bayes AUC` = sim$bayes_auc,
            Rule = r, Seed = sd_i, Fold = i, t(sc),
            check.names = FALSE, stringsAsFactors = FALSE)
        }
        cat(silver(sprintf("  seed %d | %-6s done\n", sd_i, r)))
      }
    }
    
    raw <- rbind(raw, do.call(rbind, rows))
    n_done <- n_done + 1
    saveRDS(raw, raw_file)
    write.csv(transform(raw, Rule = rule_label[Rule]),
              file.path(OUT_DIR, "k simulation raw - revision 1.csv"), row.names = FALSE)
  }
}

# ------------------------------------------------------------------------------
# 5. SUMMARY: mean over folds and seeds -> one value per dataset x rule; ranks across rules within every
#    dataset; Friedman, paired Wilcoxon (Holm) against CoverNet; mean ranks by factor and by cell.
# ------------------------------------------------------------------------------
METRICS_REPORT <- c("MCC", "P4", "IAM", "ROCAUC")
rule_order     <- names(K_RULES)
design <- unique(raw[, c("Dataset", "Cell", "n", "p", "IR", "Clusters per class")])
FACTORS <- c("n", "p", "IR", "Clusters per class")

summary_rows <- list(); factor_rows <- list(); cell_rows <- list()
for (m in METRICS_REPORT) {
  per_ds <- aggregate(raw[[m]], by = list(Dataset = raw$Dataset, Rule = raw$Rule), FUN = mean, na.rm = TRUE)
  mat <- tapply(per_ds$x, list(per_ds$Dataset, per_ds$Rule), mean)[, rule_order, drop = FALSE]
  mat <- mat[complete.cases(mat), , drop = FALSE]
  dz  <- design[match(rownames(mat), design$Dataset), ]
  
  ranks     <- t(apply(-mat, 1, rank))                 # rank 1 = best
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
  
  mr_row <- function(ii) {
    mr <- sapply(rule_order, function(v) mean(ranks[ii, v]))
    names(mr) <- unname(rule_label[rule_order])
    mr
  }
  for (f in FACTORS) {
    for (lev in sort(unique(dz[[f]]))) {
      ii <- dz[[f]] == lev
      factor_rows[[length(factor_rows) + 1]] <- cbind(
        data.frame(Metric = m, Factor = f, Level = lev, `N datasets` = sum(ii),
                   check.names = FALSE, stringsAsFactors = FALSE),
        data.frame(as.list(mr_row(ii)), check.names = FALSE))
    }
  }
  for (cl in unique(dz$Cell)) {
    ii <- dz$Cell == cl
    mr <- mr_row(ii)
    cell_rows[[length(cell_rows) + 1]] <- cbind(
      data.frame(Metric = m, Cell = cl, `N datasets` = sum(ii), check.names = FALSE,
                 stringsAsFactors = FALSE),
      data.frame(as.list(mr), check.names = FALSE),
      data.frame(`Best rule` = names(mr)[which.min(mr)], check.names = FALSE, stringsAsFactors = FALSE))
  }
}
summary_df <- do.call(rbind, summary_rows)
factor_df  <- do.call(rbind, factor_rows)
cell_df    <- do.call(rbind, cell_rows)
write.csv(summary_df, file.path(OUT_DIR, "k simulation summary - revision 1.csv"),   row.names = FALSE)
write.csv(factor_df,  file.path(OUT_DIR, "k simulation by factor - revision 1.csv"), row.names = FALSE)
write.csv(cell_df,    file.path(OUT_DIR, "k simulation by cell - revision 1.csv"),   row.names = FALSE)
print(summary_df, digits = 3)
print(factor_df,  digits = 3)

# how often each rule is the best one (lowest mean rank) across the cells
print(table(Metric = cell_df$Metric, `Best rule` = cell_df$`Best rule`))

# cost: K actually used, epochs and time per rule, by n, p and IR (averages over folds, seeds, datasets)
cost_df <- aggregate(raw[, c("Kpos", "Kneg", "Epochs", "BestEpoch", "InitSec", "TrainSec", "Runtime")],
                     by = list(Rule = raw$Rule, n = raw$n, p = raw$p, IR = raw$IR),
                     FUN = mean, na.rm = TRUE)
cost_df$Rule <- unname(rule_label[cost_df$Rule])
write.csv(cost_df, file.path(OUT_DIR, "k simulation cost - revision 1.csv"), row.names = FALSE)
print(head(cost_df, 20), digits = 3)

cat(green("\nDone.\n"))
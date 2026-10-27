# ==============================================================================
# Script: ablation family sensitivity.R   (NEW; run after run ablation.R, from the project root)
#
# Part A  Family-level sensitivity (Reviewer 3 - #7): 72 datasets -> source-dataset families.
#         Per-dataset ranks across the 7 variants are averaged WITHIN a family (raw metric values
#         are never averaged across datasets), then full vs variant is tested with a paired
#         Wilcoxon signed-rank test (Holm, 6 comparisons per metric) on the family-level ranks.
#
# Output:
#   results/ablation/ablation families - revision 1.csv, ablation family sensitivity - revision 1.csv
# ==============================================================================

OUT_DIR <- file.path("results", "ablation")    # data / tables of the ablation
.old_raw <- file.path(OUT_DIR, "ablation_raw.rds"); .new_raw <- file.path(OUT_DIR, "ablation raw - revision 1.rds")
if (!file.exists(.new_raw) && file.exists(.old_raw)) file.rename(.old_raw, .new_raw)   # file from the first run
raw     <- readRDS(.new_raw)

variants <- c("full", "no_cov", "no_rep", "no_pen", "no_aux", "init_sample", "init_randn")
safe     <- c(full = "Full", no_cov = "noCov", no_rep = "noRep", no_pen = "noPen",
              no_aux = "noAux", init_sample = "initSample", init_randn = "initRandn")
metrics  <- c(MCC = "MCC", P4 = "P4", IAM = "IAM", ROCAUC = "AUC")   # raw column -> label in the paper

stopifnot(all(variants %in% unique(raw$Variant)), all(names(metrics) %in% colnames(raw)))

# ------------------------------------------------------------------------------
# 0. Aggregation: seeds -> folds -> datasets (mean and SD over folds)
# ------------------------------------------------------------------------------
by_fold <- list(Dataset = raw$Dataset, Variant = raw$Variant, Fold = raw$Fold)
fold_lvl <- aggregate(raw[, names(metrics)], by = by_fold, FUN = mean, na.rm = TRUE)

by_ds   <- list(Dataset = fold_lvl$Dataset, Variant = fold_lvl$Variant)
ds_mean <- aggregate(fold_lvl[, names(metrics)], by = by_ds, FUN = mean, na.rm = TRUE)
ds_sd   <- aggregate(fold_lvl[, names(metrics)], by = by_ds, FUN = sd,   na.rm = TRUE)

to_wide <- function(df, col) {
  m <- tapply(df[[col]], list(df$Dataset, df$Variant), mean)[, variants, drop = FALSE]
  colnames(m) <- unname(safe[variants])
  as.data.frame(m)
}

# ------------------------------------------------------------------------------
# PART A. Family-level sensitivity
# ------------------------------------------------------------------------------
family_of <- function(nm) {
  pats <- c(BEED = "^BEED", ecoli = "^ecoli", vowel = "^vowel", segment = "^segment",
            vehicle = "^vehicle", kr_vs_k = "^kr_vs_k", winequality = "^winequality",
            yeast = "^yeast", abalone = "^abalone", kddcup = "^kddcup", poker = "^poker",
            mfeat = "^mfeat", glass = "^glass")
  for (f in names(pats)) if (grepl(pats[[f]], nm, ignore.case = TRUE)) return(f)
  nm                                                          # every other dataset is its own family
}

ds_names <- rownames(to_wide(ds_mean, names(metrics)[1]))
fam      <- vapply(ds_names, family_of, character(1))
cat(sprintf("\n%d datasets -> %d families\n", length(ds_names), length(unique(fam))))
write.csv(data.frame(Dataset = ds_names, Family = unname(fam)),
          file.path(OUT_DIR, "ablation families - revision 1.csv"), row.names = FALSE)

fam_rows <- list()
for (mt in names(metrics)) {
  mat <- as.matrix(to_wide(ds_mean, mt))
  rk  <- t(apply(-mat, 1, rank))                                          # dataset-level ranks, 1 = best
  fam_rk <- t(sapply(split(seq_len(nrow(rk)), fam),
                     function(i) colMeans(rk[i, , drop = FALSE])))        # families x variants
  
  others <- setdiff(colnames(fam_rk), "Full")
  p_raw  <- sapply(others, function(v)
    tryCatch(wilcox.test(fam_rk[, "Full"], fam_rk[, v], paired = TRUE, exact = FALSE)$p.value,
             error = function(e) NA_real_))
  p_holm <- p.adjust(p_raw, method = "holm")
  
  for (v in colnames(fam_rk)) {
    fam_rows[[length(fam_rows) + 1]] <- data.frame(
      Metric          = metrics[[mt]],
      Variant         = v,
      `N families`    = nrow(fam_rk),
      `Mean rank over families` = mean(fam_rk[, v]),                  # mean over families of the family-level rank
      `Full wins`      = if (v == "Full") NA else sum(fam_rk[, "Full"] <  fam_rk[, v]),
      `Ties`          = if (v == "Full") NA else sum(fam_rk[, "Full"] == fam_rk[, v]),
      `Full losses`    = if (v == "Full") NA else sum(fam_rk[, "Full"] >  fam_rk[, v]),
      `Holm adjusted p (Wilcoxon)` = if (v == "Full") NA else unname(p_holm[v]),
      stringsAsFactors = FALSE, check.names = FALSE)
  }
}
fam_df <- do.call(rbind, fam_rows)
write.csv(fam_df, file.path(OUT_DIR, "ablation family sensitivity - revision 1.csv"), row.names = FALSE)
print(fam_df, digits = 3)

cat("\nDone.\n")
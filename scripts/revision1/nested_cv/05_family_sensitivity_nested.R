# ==============================================================================
# Script: scripts/revision1/nested_cv/05_family_sensitivity_nested.R   (NEW FILE, put it next to run.R)
# Purpose: reviewer 3 comment 7. The 72 datasets are variants of 30 source datasets (families: vowel, ecoli, segment, ...).
#          The comparison of CoverNET with its 18 rivals (nested CV, "tuned RF and SVM nested" scenario, the main evaluation)
#          is repeated with ONE unit per family.
# Tests  : the same as 'scripts/revision1/nested_cv/02_nested_cv_summary.R': paired Wilcoxon (normal approximation, zeros dropped) and Conover
#          post hoc Friedman, Holm over all pairs of the 19 methods. Friedman test p value as well.
# Analyses per metric (IAM, MCC, P4, AUC):
#   72 datasets                    : the analysis of the main evaluation (control, must equal the nested summary)
#   30 families, mean performance  : metric values averaged WITHIN a family (versions of the same source data)
#   30 families, mean rank         : the per-dataset ranks averaged within a family
# Raw values are never averaged across families.
# Output : results/nested CV - revision 1/family sensitivity - revision 1.xlsx (sheets: Families, Summary, P values)
# ==============================================================================

library(openxlsx)

NESTED  <- "results/nested CV - revision 1/results.xlsx"
PAPER   <- "results/results combined.xlsx"
METRICS <- c("IAM", "MCC", "P4", "ROCAUC")
mlab    <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", ROCAUC = "AUC")

nested <- read.xlsx(NESTED, sheet = "Means"); names(nested)[names(nested) == "CCCDNNET"] <- "CoverNET"
paper  <- read.xlsx(PAPER,  sheet = "Means")
ds     <- unique(nested$Dataset)
methods <- c("RFtuned", "RUSRFtuned", "SMOTERFtuned", "SVMtuned", "SMOTESVMtuned", "NaiveBayes", "CART", "KNN", "LogReg",
             "NNET", "BalancedNNET", "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "PNN", "SkewPNN", "CoverNET")

get_perf <- function(metric) {
  a <- nested[nested$Metric == metric, ]; a <- a[match(ds, a$Dataset), ]
  b <- paper[paper$Metric == metric, ];   b <- b[match(ds, b$Dataset), ]
  x <- sapply(methods, function(m) if (m %in% names(a)) a[[m]] else b[[m]])
  matrix(x, nrow = length(ds), dimnames = list(ds, methods))
}

# families: datasets that are variants of the same source data (every other dataset is its own family)
family_of <- function(nm) {
  pats <- c(BEED = "^BEED", ecoli = "^ecoli", vowel = "^vowel", segment = "^segment", vehicle = "^vehicle",
            kr_vs_k = "^kr_vs_k", winequality = "^winequality", yeast = "^yeast", abalone = "^abalone",
            kddcup = "^kddcup", poker = "^poker", mfeat = "^mfeat", glass = "^glass")
  for (f in names(pats)) if (grepl(pats[[f]], nm, ignore.case = TRUE)) return(f)
  nm
}
families <- data.frame(Dataset = ds, Family = vapply(ds, family_of, character(1)), row.names = NULL)
cat(sprintf("%d datasets -> %d families\n", nrow(families), length(unique(families$Family))))

wilcox_p <- function(a, b) if (all(a == b)) 1 else wilcox.test(a, b, paired = TRUE, exact = FALSE, correct = FALSE)$p.value

# x: units x methods, larger = better. Friedman p, Holm Wilcoxon and Holm Conover of CoverNET against every rival
run_tests <- function(x) {
  r  <- t(apply(-x, 1, rank)); b <- nrow(x); k <- ncol(x); Rs <- colSums(r)
  A  <- sum(r^2); C <- b * k * (k + 1)^2 / 4
  T1 <- (k - 1) * sum((Rs - b * (k + 1) / 2)^2) / (A - C)                   # Friedman statistic (ties corrected)
  S  <- sqrt(2 * b * (A - C) / ((b - 1) * (k - 1)))
  pr <- t(combn(colnames(x), 2))
  pw <- p.adjust(apply(pr, 1, function(p) wilcox_p(x[, p[1]], x[, p[2]])), "holm")
  pc <- p.adjust(2 * pt(-abs(Rs[pr[, 1]] - Rs[pr[, 2]]) / S, (b - 1) * (k - 1)), "holm")
  i  <- which(pr[, 1] == "CoverNET" | pr[, 2] == "CoverNET")
  list(friedman = pchisq(T1, k - 1, lower.tail = FALSE),
       p = data.frame(Competitor = ifelse(pr[i, 1] == "CoverNET", pr[i, 2], pr[i, 1]), Wilcoxon = pw[i], Conover = pc[i]))
}
within_family <- function(M, fam) t(sapply(split(seq_len(nrow(M)), fam), function(i) colMeans(M[i, , drop = FALSE])))

summary_rows <- p_rows <- list()
for (mt in METRICS) {
  P   <- get_perf(mt)
  R   <- t(apply(-P, 1, rank))                                              # rank 1 = best, per dataset
  fam <- families$Family
  analyses <- list(`72 datasets`                   = P,
                   `30 families, mean performance` = within_family(P, fam),
                   `30 families, mean rank`        = -within_family(R, fam))  # lower rank = better, so negate
  for (a in names(analyses)) {
    out <- run_tests(analyses[[a]])
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      Metric = mlab[[mt]], Analysis = a, Units = nrow(analyses[[a]]), `Friedman p` = out$friedman,
      `Wilcoxon significant (of 18)` = sum(out$p$Wilcoxon < 0.05), `Conover significant (of 18)` = sum(out$p$Conover < 0.05),
      `Wilcoxon not significant` = paste(out$p$Competitor[out$p$Wilcoxon >= 0.05], collapse = ", "),
      `Conover not significant`  = paste(out$p$Competitor[out$p$Conover  >= 0.05], collapse = ", "),
      check.names = FALSE, stringsAsFactors = FALSE)
    p_rows[[length(p_rows) + 1]] <- data.frame(Metric = mlab[[mt]], Analysis = a, out$p)
  }
}
summary_df <- do.call(rbind, summary_rows)
p_df       <- do.call(rbind, p_rows)
print(summary_df[, 1:6], digits = 3)
write.xlsx(list(Families = families, Summary = summary_df, `P values` = p_df),
           "results/nested CV - revision 1/family sensitivity - revision 1.xlsx")
cat("\nCheck: the '72 datasets' rows must match the nested summary (Holm Wilcoxon < 0.05 against 14 / 13 / 13 / 7 rivals for IAM / MCC / P4 / AUC).\n")

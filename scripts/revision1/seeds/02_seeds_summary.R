# ==============================================================================
# Script: scripts/revision1/seeds/02_seeds_summary.R   (NEW FILE, put it next to run.R)
# Purpose: reviewer 3 comment 11. Stability of the neural comparison over the N seeds of 'scripts/revision1/seeds/01_run_seeds.R'.
#   For every seed the 10 neural methods are ranked per dataset (8 re-evaluated methods with that seed, PNN and SkewPNN from
#   the nested CV, they are deterministic). No raw metric is averaged over datasets: ranks, W/T/L-free Holm Wilcoxon only.
# Input  : results/seeds - revision 1/results long.csv, results/nested CV - revision 1/results.xlsx
# Output : results/seeds - revision 1/seeds summary - revision 1.xlsx
#            sheet "Mean ranks"  : per metric and method, mean rank over the seeds, SD over the seeds, min, max
#            sheet "CoverNET"    : per metric, CoverNET position in every seed and the number of rivals (of 9) it beats
#                                  significantly (paired Wilcoxon, Holm over the 45 pairs) in every seed
#          results/Figures/seeds mean rank - revision 1.png  (mean rank of every method in each seed)
# ==============================================================================

library(openxlsx)
library(ggplot2)

LONG    <- "results/seeds - revision 1/results long.csv"
NESTED  <- "results/nested CV - revision 1/results.xlsx"
METRICS <- c("IAM", "MCC", "P4", "ROCAUC")
mlab    <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", ROCAUC = "AUC")
random  <- c("NNET", "BalancedNNET", "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "CCCDNNET")
fixed   <- c("PNN", "SkewPNN")
label   <- c(NNET = "NN", BalancedNNET = "BalNN", SMOTENNET = "SMOTE + NN", FocalNNET = "FocalNN", FocalSDNNET = "FocalSDNN",
             LDAMNNET = "LDAMNN", MMNNET = "MMNN", PNN = "PNN", SkewPNN = "SkewPNN", CCCDNNET = "CoverNET")

long   <- read.csv(LONG)
nested <- read.xlsx(NESTED, sheet = "Means")
ds     <- sort(unique(long$Dataset))
seeds  <- sort(unique(long$Seed))
stopifnot(length(ds) == 72, !anyNA(long))

wilcox_p <- function(a, b) if (all(a == b)) 1 else wilcox.test(a, b, paired = TRUE, exact = FALSE, correct = FALSE)$p.value

perf <- function(metric, s) {                                  # datasets x 10 methods for one seed
  a <- long[long$Seed == s, ]
  x <- sapply(random, function(m) { b <- a[a$Method == m, ]; b[[metric]][match(ds, b$Dataset)] })
  n <- nested[nested$Metric == metric, ]; n <- n[match(ds, n$Dataset), ]
  x <- cbind(x, sapply(fixed, function(m) n[[m]]))
  rownames(x) <- ds
  x
}

rank_rows <- cn_rows <- list()
for (mt in METRICS) {
  R <- sapply(seeds, function(s) colMeans(t(apply(-perf(mt, s), 1, rank))))          # methods x seeds
  rank_rows[[mt]] <- data.frame(Metric = mlab[[mt]], Method = label[rownames(R)], MeanRank = round(rowMeans(R), 2),
                                SD = round(apply(R, 1, sd), 3), Min = round(apply(R, 1, min), 2), Max = round(apply(R, 1, max), 2))
  pos <- apply(R, 2, function(r) rank(r, ties.method = "min")["CCCDNNET"])
  sig <- sapply(seeds, function(s) {
    x  <- perf(mt, s); rv <- setdiff(colnames(x), "CCCDNNET")
    pr <- t(combn(colnames(x), 2))
    p  <- p.adjust(apply(pr, 1, function(q) wilcox_p(x[, q[1]], x[, q[2]])), "holm")
    sum(p[pr[, 1] == "CCCDNNET" | pr[, 2] == "CCCDNNET"] < 0.05)
  })
  cn_rows[[mt]] <- data.frame(Metric = mlab[[mt]], Seed = seeds, CoverNETMeanRank = round(R["CCCDNNET", ], 2),
                              Position = as.integer(pos), SignificantOf9 = sig)
  cat(sprintf("%-6s CoverNET mean rank %.2f (SD %.2f) | position 1 in %d of %d seeds | significant wins of 9: %d-%d\n",
              mlab[[mt]], mean(R["CCCDNNET", ]), sd(R["CCCDNNET", ]), sum(pos == 1), length(seeds), min(sig), max(sig)))
}
rank_tab <- do.call(rbind, rank_rows); cn_tab <- do.call(rbind, cn_rows)
rownames(rank_tab) <- rownames(cn_tab) <- NULL
write.xlsx(list("Mean ranks" = rank_tab, "CoverNET" = cn_tab), "results/seeds - revision 1/seeds summary - revision 1.xlsx")

# figure: mean rank of every method in each seed (CoverNET in red)
fig <- do.call(rbind, lapply(METRICS, function(mt) do.call(rbind, lapply(seeds, function(s) {
  r <- colMeans(t(apply(-perf(mt, s), 1, rank)))
  data.frame(Metric = mlab[[mt]], Seed = s, Method = label[names(r)], MeanRank = as.numeric(r))
}))))
fig$Metric <- factor(fig$Metric, levels = mlab)
ord <- names(sort(tapply(fig$MeanRank[fig$Metric == "MCC"], fig$Method[fig$Metric == "MCC"], mean), decreasing = TRUE))
fig$Method <- factor(fig$Method, levels = ord)
plt <- ggplot(fig, aes(x = MeanRank, y = Method, color = Method == "CoverNET")) +
  geom_point(alpha = 0.6, size = 1.8) +
  scale_color_manual(values = c("FALSE" = "black", "TRUE" = "red"), guide = "none") +
  facet_wrap(~ Metric, nrow = 1) +
  labs(x = "Mean rank over the 72 datasets (one point per seed, lower is better)", y = NULL) +
  theme_bw() + theme(strip.background = element_rect(fill = NA, color = NA))
dir.create("results/Figures", showWarnings = FALSE)
ggsave("results/Figures/seeds mean rank - revision 1.png", plot = plt, width = 12, height = 4, dpi = 300)

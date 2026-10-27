# ==============================================================================
# Script: scripts/revision1/nested_cv/08_matched_heads_summary.R   (NEW FILE, put it next to run.R)
# Purpose: matched-head comparison (reviewer 1 comment 2, reviewer 3 comment 3). Same hidden layers, same BCE loss, same
#          architecture grid, nested CV; only the output layer differs. No raw metric is averaged over datasets.
# Methods: NNET (linear head), MLPheadNNET, RBFheadNNET, CCLBCENNET (CoverNET head, classification term only), CoverNET
# Input  : results/nested CV - revision 1/results.xlsx
# Output : results/nested CV - revision 1/matched heads - revision 1.xlsx
#            sheet "Mean ranks"  : mean rank (1 = best) of the 5 methods per metric
#            sheet "CoverNET vs" : W/T/L of CoverNET against each head, Holm Wilcoxon (over the 10 pairs of a metric)
#            sheet "All pairs"   : Holm Wilcoxon and Holm Conover of all 10 pairs
#          results/Figures/matched heads mean rank - revision 1.png
# ==============================================================================

library(openxlsx)
library(ggplot2)

NESTED  <- "results/nested CV - revision 1/results.xlsx"
METRICS <- c("IAM", "MCC", "P4", "ROCAUC")
METHODS <- c("NNET", "MLPheadNNET", "RBFheadNNET", "CCLBCENNET", "CCCDNNET")
label   <- c(NNET = "Linear head (NN)", MLPheadNNET = "MLP head", RBFheadNNET = "RBF head",
             CCLBCENNET = "CCL head, BCE only", CCCDNNET = "CoverNET")
mlab    <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", ROCAUC = "AUC")

nested <- read.xlsx(NESTED, sheet = "Means")
ds     <- unique(nested$Dataset)
get_perf <- function(metric) {
  a <- nested[nested$Metric == metric, ]; a <- a[match(ds, a$Dataset), ]
  x <- as.matrix(a[, METHODS]); rownames(x) <- ds
  if (anyNA(x)) stop("missing results for ", metric)
  x
}

wilcox_p <- function(a, b) if (all(a == b)) 1 else wilcox.test(a, b, paired = TRUE, exact = FALSE, correct = FALSE)$p.value

ranks <- vs <- pairs <- list()
for (mt in METRICS) {
  x  <- get_perf(mt)
  r  <- t(apply(-x, 1, rank)); b <- nrow(x); k <- ncol(x); Rs <- colSums(r)
  S  <- sqrt(2 * b * (sum(r^2) - b * k * (k + 1)^2 / 4) / ((b - 1) * (k - 1)))
  pr <- t(combn(METHODS, 2))
  pw <- p.adjust(apply(pr, 1, function(p) wilcox_p(x[, p[1]], x[, p[2]])), "holm")
  pc <- p.adjust(2 * pt(-abs(Rs[pr[, 1]] - Rs[pr[, 2]]) / S, (b - 1) * (k - 1)), "holm")
  ranks[[mt]] <- data.frame(Metric = mlab[[mt]], Method = label[METHODS], MeanRank = round(colMeans(r), 2))
  pairs[[mt]] <- data.frame(Metric = mlab[[mt]], A = label[pr[, 1]], B = label[pr[, 2]], HolmWilcoxon = pw, HolmConover = pc)
  rv <- setdiff(METHODS, "CCCDNNET")
  p  <- sapply(rv, function(m) pw[(pr[, 1] == m & pr[, 2] == "CCCDNNET") | (pr[, 2] == m & pr[, 1] == "CCCDNNET")])
  vs[[mt]] <- data.frame(Metric = mlab[[mt]], Head = label[rv],
                         Win  = sapply(rv, function(m) sum(x[, "CCCDNNET"] > x[, m])),
                         Tie  = sapply(rv, function(m) sum(x[, "CCCDNNET"] == x[, m])),
                         Loss = sapply(rv, function(m) sum(x[, "CCCDNNET"] < x[, m])),
                         HolmWilcoxon = p)
}
ranks <- do.call(rbind, ranks); vs <- do.call(rbind, vs); pairs <- do.call(rbind, pairs)
rownames(ranks) <- rownames(vs) <- rownames(pairs) <- NULL

write.xlsx(list("Mean ranks" = ranks, "CoverNET vs" = vs, "All pairs" = pairs),
           "results/nested CV - revision 1/matched heads - revision 1.xlsx")
print(ranks); print(vs)

dir.create("results/Figures", showWarnings = FALSE)
ranks$Metric <- factor(ranks$Metric, levels = mlab)
ranks$Method <- factor(ranks$Method, levels = rev(label[METHODS]))
plt <- ggplot(ranks, aes(x = MeanRank, y = Method, color = Method == "CoverNET")) +
  geom_segment(aes(xend = 1, yend = Method), linewidth = 0.4) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c("FALSE" = "black", "TRUE" = "red"), guide = "none") +
  facet_wrap(~ Metric, nrow = 1) +
  labs(x = "Mean rank (lower is better)", y = NULL) +
  theme_bw() + theme(strip.background = element_rect(fill = NA, color = NA))
ggsave("results/Figures/matched heads mean rank - revision 1.png", plot = plt, width = 12, height = 3.5, dpi = 300)

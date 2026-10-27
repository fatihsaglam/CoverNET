# ==============================================================================
# Script: scripts/revision1/nested_cv/07_stratified_plots.R   (NEW FILE, put it next to run.R; the original plots.R is untouched)
# Purpose: reviewer 3 comment 6 (inconsistent stratified counts). Sections 5.3-5.6 and Figures 4-7 regenerated on the nested CV
#          results ("tuned RF and SVM nested" scenario, 19 methods), with mutually exclusive quartile bins.
#   Bins   : quartiles of the 72 dataset-level values, cut into (q0, q1], (q1, q2], (q2, q3], (q3, q4] (the first bin also
#            contains q0). A dataset on a boundary belongs to the lower bin only, so the counts always add up to 72.
#            Bin labels show the observed range of the datasets in the bin, and the number of datasets (n = ...).
#   Ranks  : per dataset and metric, among the 19 methods (1 = best, average ties), as in the paper.
# Input  : results/nested CV - revision 1/results.xlsx, results/results combined.xlsx, results/dataset info.xlsx
# Output : results/Figures/rank vs n box nested - revision 1.png, ... p, ... IR, ... pn
#          results/nested CV - revision 1/stratified bins - revision 1.xlsx
#            sheet "Bin counts" : variable, bin, observed range, number of datasets (sums to 72 for every variable)
#            sheet "Mean ranks" : variable, bin, metric, method, mean rank, position (CoverNET highlighted by the column Position)
# ==============================================================================

library(openxlsx)
library(ggplot2)

NESTED  <- "results/nested CV - revision 1/results.xlsx"
PAPER   <- "results/results combined.xlsx"
METRICS <- c("IAM", "MCC", "P4", "ROCAUC")
mlab    <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", ROCAUC = "AUC")

nested <- read.xlsx(NESTED, sheet = "Means"); names(nested)[names(nested) == "CCCDNNET"] <- "CoverNET"
paper  <- read.xlsx(PAPER,  sheet = "Means")
info   <- read.xlsx("results/dataset info.xlsx")
ds     <- unique(nested$Dataset)
info   <- info[match(ds, info$dataset_name), ]
stopifnot(!anyNA(info$n), nrow(info) == 72)

methods <- c("RFtuned", "RUSRFtuned", "SMOTERFtuned", "SVMtuned", "SMOTESVMtuned", "NaiveBayes", "CART", "KNN", "LogReg",
             "NNET", "BalancedNNET", "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "PNN", "SkewPNN", "CoverNET")
label <- c(NaiveBayes = "NB", CART = "DT", KNN = "KNN", LogReg = "LogReg", NNET = "NN", BalancedNNET = "BalNN",
           SMOTENNET = "SMOTE + NN", FocalNNET = "FocalNN", FocalSDNNET = "FocalSDNN", LDAMNNET = "LDAMNN", MMNNET = "MMNN",
           PNN = "PNN", SkewPNN = "SkewPNN", CoverNET = "CoverNET", RFtuned = "Tuned RF", RUSRFtuned = "Tuned RUS + RF",
           SMOTERFtuned = "Tuned SMOTE + RF", SVMtuned = "Tuned SVM", SMOTESVMtuned = "Tuned SMOTE + SVM")

get_perf <- function(metric) {
  a <- nested[nested$Metric == metric, ]; a <- a[match(ds, a$Dataset), ]
  b <- paper[paper$Metric == metric, ];   b <- b[match(ds, b$Dataset), ]
  x <- sapply(methods, function(m) if (m %in% names(a)) a[[m]] else b[[m]])
  matrix(x, nrow = length(ds), dimnames = list(ds, methods))
}

# per dataset, metric and method: rank among the 19 methods
rk <- do.call(rbind, lapply(METRICS, function(mt) {
  r <- t(apply(-get_perf(mt), 1, rank))
  data.frame(Dataset = rep(ds, times = length(methods)), Metric = mlab[[mt]],
             Method = rep(label[methods], each = length(ds)), Rank = as.vector(r))
}))
rk$Method <- factor(rk$Method, levels = label[methods])
rk$Metric <- factor(rk$Metric, levels = mlab)

# mutually exclusive quartile bins at the dataset level
make_bins <- function(x, digits) {
  q <- quantile(x, c(0, 0.25, 0.5, 0.75, 1))
  b <- as.integer(cut(x, q, include.lowest = TRUE))                  # right closed, first bin includes the minimum
  stopifnot(!anyNA(b))
  rng <- sapply(1:4, function(i) sprintf("[%s, %s]", format(round(min(x[b == i]), digits), nsmall = 0, trim = TRUE),
                                         format(round(max(x[b == i]), digits), nsmall = 0, trim = TRUE)))
  cnt <- as.integer(table(factor(b, 1:4)))
  stopifnot(sum(cnt) == length(x))                                   # the counts add up to 72
  list(bin = factor(paste0(rng, "\n(n = ", cnt, ")"), levels = paste0(rng, "\n(n = ", cnt, ")"))[b], range = rng, count = cnt)
}
vars <- list(n  = list(col = "n",              lab = "Sample size (n)",         digits = 0),
             p  = list(col = "p",              lab = "Number of features (p)",  digits = 0),
             IR = list(col = "IR",             lab = "Imbalance ratio (IR)",    digits = 1),
             pn = list(col = "dimensionality", lab = "p/n",                     digits = 3))

counts <- ranks_tab <- list()
dir.create("results/Figures", showWarnings = FALSE)
for (v in names(vars)) {
  bins <- make_bins(info[[vars[[v]]$col]], vars[[v]]$digits)
  counts[[v]] <- data.frame(Variable = vars[[v]]$lab, Bin = 1:4, ObservedRange = bins$range, Datasets = bins$count)
  d <- rk; d$bin <- bins$bin[match(d$Dataset, ds)]

  mr <- aggregate(Rank ~ bin + Metric + Method, data = d, FUN = mean)
  mr$Position <- ave(mr$Rank, mr$bin, mr$Metric, FUN = function(r) rank(r, ties.method = "min"))
  ranks_tab[[v]] <- data.frame(Variable = vars[[v]]$lab, Bin = as.character(mr$bin), Metric = mr$Metric, Method = mr$Method,
                               MeanRank = round(mr$Rank, 2), Position = mr$Position)

  plt <- ggplot(d, aes(x = bin, y = Rank, fill = Method)) +
    stat_boxplot(geom = "errorbar", width = 0.5, linewidth = 0.2, position = position_dodge(width = 0.9)) +
    geom_boxplot(outlier.size = 0.5, linewidth = 0.2, width = 0.9, position = position_dodge(width = 0.9)) +
    facet_wrap(. ~ Metric, nrow = 2, ncol = 2) +
    geom_vline(xintercept = seq(1.5, 3.5, 1), linetype = 2, linewidth = 0.3, color = "gray60") +
    scale_fill_manual(values = rev(ggsci::pal_d3("category20")(19))) +
    scale_x_discrete(expand = c(0.02, 0.02)) +
    labs(x = vars[[v]]$lab) +
    theme_bw() + guides(fill = guide_legend(nrow = 2)) +
    theme(legend.position = "top", legend.key.height = unit(0.3, "cm"),
          axis.text.x = element_text(size = 12, color = "black"), axis.text.y = element_text(size = 12, color = "black"),
          strip.background = element_rect(fill = NA, colour = NA), strip.text = element_text(size = 12, color = "black"),
          text = element_text(size = 12, color = "black"))
  ggsave(sprintf("results/Figures/rank vs %s box nested - revision 1.png", v), plot = plt, height = 5, width = 12, dpi = 300)
}
counts     <- do.call(rbind, counts)
ranks_tab  <- do.call(rbind, ranks_tab)
write.xlsx(list("Bin counts" = counts, "Mean ranks" = ranks_tab), "results/nested CV - revision 1/stratified bins - revision 1.xlsx")

print(counts)
# on screen: CoverNET position per bin and metric (1 = best mean rank among the 19 methods)
cn <- ranks_tab[ranks_tab$Method == "CoverNET", ]
print(reshape(cn[, c("Variable", "Bin", "Metric", "Position")], idvar = c("Variable", "Bin"), timevar = "Metric", direction = "wide"))

# ==============================================================================
# Script: scripts/revision1/nested_cv/04_non_saturated_datasets.R   (NEW FILE, put it next to run.R)
# Purpose: secondary analysis of the nested CV (reviewer 3, comment 4). Same tests as 'scripts/revision1/nested_cv/02_nested_cv_summary.R',
#          but only on the datasets that are NOT saturated: the best of the 24 methods (19 paper methods + 5 tuned RF and
#          SVM versions, nested) has ROCAUC < SAT_LIMIT. The rule is fixed in advance and does not use CoverNET.
# Scenario: "tuned RF and SVM nested" (the main evaluation). No raw metric is averaged over datasets.
# Output (results/nested CV - revision 1/):
#   nested CV non saturated datasets - revision 1.csv     datasets used and best ROCAUC
#   nested CV non saturated mean ranks - revision 1.csv   mean rank of every method and metric
#   nested CV non saturated CoverNET vs rivals - revision 1.csv   W/T/L, rank difference, Holm Wilcoxon, Holm Conover
#   results/Figures/Bayesian comparison boxplot non saturated - revision 1.png
#   results/Bradley Terry model results non saturated - revision 1.xlsx
# ==============================================================================

library(openxlsx)
library(bbtcomp)
library(ggplot2)
library(ggtext)

NESTED    <- "results/nested CV - revision 1/results.xlsx"
PAPER     <- "results/results combined.xlsx"
OUT_DIR   <- "results/nested CV - revision 1"
METRICS   <- c("IAM", "MCC", "P4", "ROCAUC")
SAT_LIMIT <- 0.99

nested <- read.xlsx(NESTED, sheet = "Means"); names(nested)[names(nested) == "CCCDNNET"] <- "CoverNET"
paper  <- read.xlsx(PAPER,  sheet = "Means")
nsd    <- read.xlsx(NESTED, sheet = "SDs");   names(nsd)[names(nsd) == "CCCDNNET"] <- "CoverNET"
psd    <- read.xlsx(PAPER,  sheet = "SDs")
ds_all <- unique(nested$Dataset)

tuned_nn  <- c("NNET", "BalancedNNET", "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "PNN", "SkewPNN", "CoverNET")
classical <- c("RF", "RUS_RF", "SMOTE_RF", "SVMRadial", "SMOTE_SVM")
tuned_cl  <- c("RFtuned", "RUSRFtuned", "SMOTERFtuned", "SVMtuned", "SMOTESVMtuned")
other     <- c("NaiveBayes", "CART", "KNN", "LogReg")
methods   <- c(tuned_cl, other, tuned_nn)                                     # the 19 methods of the main evaluation

get_perf <- function(metric, ms, ds, nst = nested, ppr = paper) {
  a <- nst[nst$Metric == metric, ]; a <- a[match(ds, a$Dataset), ]
  b <- ppr[ppr$Metric == metric, ]; b <- b[match(ds, b$Dataset), ]
  matrix(sapply(ms, function(m) if (m %in% names(a)) a[[m]] else b[[m]]), nrow = length(ds), dimnames = list(ds, ms))
}

# saturation rule: best of all 24 methods (19 + the 5 untuned RF and SVM versions), nested ROCAUC
all24 <- get_perf("ROCAUC", c(methods, classical), ds_all)
best  <- apply(all24, 1, max)
ds    <- ds_all[best < SAT_LIMIT]
cat(sprintf("Non saturated datasets: %d of %d (best ROCAUC < %.2f)\n", length(ds), length(ds_all), SAT_LIMIT))
write.csv(data.frame(Dataset = ds_all, BestROCAUC = round(best, 4), Used = best < SAT_LIMIT),
          file.path(OUT_DIR, "nested CV non saturated datasets - revision 1.csv"), row.names = FALSE)

wilcox_p <- function(a, b) if (all(a == b)) 1 else wilcox.test(a, b, paired = TRUE, exact = FALSE, correct = FALSE)$p.value
analyse <- function(x) {
  r  <- t(apply(-x, 1, rank)); b <- nrow(x); k <- ncol(x); Rs <- colSums(r)
  S  <- sqrt(2 * b * (sum(r^2) - b * k * (k + 1)^2 / 4) / ((b - 1) * (k - 1)))
  pr <- t(combn(colnames(x), 2))
  list(mean_rank = colMeans(r),
       pairs = data.frame(a = pr[, 1], b = pr[, 2],
                          holm_wilcoxon = p.adjust(apply(pr, 1, function(p) wilcox_p(x[, p[1]], x[, p[2]])), "holm"),
                          holm_conover  = p.adjust(2 * pt(-abs(Rs[pr[, 1]] - Rs[pr[, 2]]) / S, (b - 1) * (k - 1)), "holm")))
}

ranks <- rivals <- list()
for (mt in METRICS) {
  x   <- get_perf(mt, methods, ds)
  res <- analyse(x)
  ranks[[mt]] <- data.frame(Metric = mt, Method = colnames(x), MeanRank = round(res$mean_rank, 3))
  p  <- res$pairs[res$pairs$a == "CoverNET" | res$pairs$b == "CoverNET", ]
  rv <- ifelse(p$a == "CoverNET", p$b, p$a)
  rivals[[mt]] <- data.frame(
    Metric = mt, Rival = rv,
    Win  = sapply(rv, function(m) sum(x[, "CoverNET"] > x[, m])),
    Tie  = sapply(rv, function(m) sum(x[, "CoverNET"] == x[, m])),
    Loss = sapply(rv, function(m) sum(x[, "CoverNET"] < x[, m])),
    RankDiff = round(res$mean_rank[rv] - res$mean_rank["CoverNET"], 3),
    HolmWilcoxon = p$holm_wilcoxon, HolmConover = p$holm_conover)
  r <- ranks[[mt]]; v <- rivals[[mt]]
  cat(sprintf("%-6s CoverNET mean rank %.2f (position %d of %d) | Holm Wilcoxon < 0.05: %d of %d | Holm Conover < 0.05: %d of %d\n",
              mt, r$MeanRank[r$Method == "CoverNET"], rank(r$MeanRank, ties.method = "min")[r$Method == "CoverNET"], nrow(r),
              sum(v$HolmWilcoxon < 0.05), nrow(v), sum(v$HolmConover < 0.05), nrow(v)))
}
write.csv(do.call(rbind, ranks),  file.path(OUT_DIR, "nested CV non saturated mean ranks - revision 1.csv"), row.names = FALSE)
write.csv(do.call(rbind, rivals), file.path(OUT_DIR, "nested CV non saturated CoverNET vs rivals - revision 1.csv"), row.names = FALSE)

# ------------------------------------------------------------------------------
# Bayesian Bradley-Terry comparison (same code as the nested summary script, on the non saturated datasets)
# ------------------------------------------------------------------------------
dir.create("results/Figures", showWarnings = FALSE)
label <- c(RF = "RF", NaiveBayes = "NB", CART = "DT", SVMRadial = "SVM", KNN = "KNN", LogReg = "LogReg",
           SMOTE_RF = "SMOTE + RF", RUS_RF = "RUS + RF", SMOTE_SVM = "SMOTE + SVM", NNET = "NN", BalancedNNET = "BalNN",
           SMOTENNET = "SMOTE + NN", FocalNNET = "FocalNN", FocalSDNNET = "FocalSDNN", LDAMNNET = "LDAMNN", MMNNET = "MMNN",
           PNN = "PNN", SkewPNN = "SkewPNN", CoverNET = "CoverNET", RFtuned = "Tuned RF", RUSRFtuned = "Tuned RUS + RF",
           SMOTERFtuned = "Tuned SMOTE + RF", SVMtuned = "Tuned SVM", SMOTESVMtuned = "Tuned SMOTE + SVM")
mlab     <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", ROCAUC = "AUC")
ROPE     <- c(0.45, 0.55)
fill_map <- c("CoverNET loses" = "red", "Practically equal" = "orange", "CoverNET wins" = "darkgreen")

fits <- lapply(setNames(METRICS, METRICS), function(mt) {
  x  <- data.frame(Dataset = ds, get_perf(mt, methods, ds), check.names = FALSE); colnames(x) <- c("Dataset", label[methods])
  sd <- as.data.frame(get_perf(mt, methods, ds, nst = nsd, ppr = psd)); colnames(sd) <- label[methods]
  bbtcomp(tabx = x, tabsd = sd)
})
lookup <- c(setNames(label[methods], label[methods]), setNames(label[methods], make.names(label[methods])))
draws <- do.call(rbind, lapply(METRICS, function(mt) {
  pw <- as.data.frame(bbtcomp:::get_pwin(fits[[mt]]))
  pw <- pw[, grepl("CoverNET", colnames(pw), ignore.case = TRUE), drop = FALSE]
  do.call(rbind, lapply(colnames(pw), function(cn) {
    side <- unname(lookup[strsplit(cn, "\\s*>\\s*")[[1]]])
    data.frame(Metric = mlab[[mt]], Rival = side[side != "CoverNET"], Win = if (side[1] == "CoverNET") pw[[cn]] else 1 - pw[[cn]])
  }))
}))
if (length(unique(draws$Rival)) != length(methods) - 1) stop("bbtcomp lost a method, check the columns of tabx")

d <- do.call(rbind, lapply(split(draws, list(draws$Metric, draws$Rival), drop = TRUE), function(g)
  data.frame(Metric = g$Metric[1], Rival = g$Rival[1], ymin = min(g$Win), lower = unname(quantile(g$Win, 0.025)),
             middle = median(g$Win), upper = unname(quantile(g$Win, 0.975)), ymax = max(g$Win))))
d$result <- factor(ifelse(d$middle > ROPE[2], "CoverNET wins", ifelse(d$middle < ROPE[1], "CoverNET loses", "Practically equal")),
                   levels = names(fill_map))
d$Metric   <- factor(d$Metric, levels = mlab)
d          <- d[order(d$Metric, -d$middle), ]
d$order_id <- ave(d$middle, d$Metric, FUN = seq_along)
d$x_key    <- factor(paste(d$Metric, d$order_id, sep = "___"), levels = paste(d$Metric, d$order_id, sep = "___"))
label_lookup <- setNames(paste0("<span style='color:", fill_map[as.character(d$result)], "'>", d$Rival, "</span>"), d$x_key)

plt <- ggplot(d, aes(x = x_key)) +
  geom_errorbar(aes(ymin = ymin, ymax = ymax), linewidth = 0.25, width = 0.5) +
  geom_boxplot(aes(ymin = ymin, lower = lower, middle = middle, upper = upper, ymax = ymax, fill = result),
               stat = "identity", linewidth = 0.25) +
  scale_fill_manual(name = NULL, values = fill_map, limits = names(fill_map)) +
  geom_hline(yintercept = 0.5, color = "red", linetype = 1) +
  geom_hline(yintercept = ROPE, color = "red", linetype = 2) +
  facet_wrap(~ Metric, nrow = 1, scales = "free_x") +
  scale_x_discrete(name = "", labels = label_lookup) +
  scale_y_continuous(name = "Posterior win probability of CoverNET", breaks = seq(0, 1, 0.1)) +
  theme_bw() +
  theme(axis.text.x = element_markdown(angle = 90, hjust = 1, vjust = 0.5), legend.position = "bottom",
        strip.background = element_rect(fill = NA, color = NA))
ggsave("results/Figures/Bayesian comparison boxplot non saturated - revision 1.png", plot = plt, width = 16, height = 6, dpi = 300)

prob <- do.call(rbind, lapply(METRICS, function(mt) data.frame(Metric = mlab[[mt]], table_pwin(fits[[mt]], control = "CoverNET"))))
write.xlsx(list(probTable = prob, posterior = d[, c("Metric", "Rival", "ymin", "lower", "middle", "upper", "ymax", "result")]),
           "results/Bradley Terry model results non saturated - revision 1.xlsx")

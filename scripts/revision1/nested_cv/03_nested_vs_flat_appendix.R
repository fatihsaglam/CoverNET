# ==============================================================================
# Script: scripts/revision1/nested_cv/03_nested_vs_flat_appendix.R   (NEW FILE, put it next to run.R)
# Purpose: appendix material for the nested CV (reviewer 3, comment 4): how much flat CV (tuning and testing on the same
#          folds) overstates the tuned methods, and how the ranking changes. No raw metric is averaged over datasets.
# Input  : the csv files of 'scripts/revision1/nested_cv/02_nested_cv_summary.R' (run it first)
# Output : results/nested CV - revision 1/nested CV appendix tables - revision 1.xlsx
#            sheet "Nested vs flat"  : per tuned neural method and metric, number of datasets (of 72) where the nested
#                                      score is lower / equal / higher than the flat CV score, Wilcoxon p (Holm over the
#                                      10 methods of each metric)
#            sheet "Mean ranks"      : mean rank and position of the 19 methods of the paper, flat CV and nested CV
#          results/Figures/nested vs flat counts - revision 1.png   (stacked bars of the three counts)
#          results/Figures/mean rank flat vs nested - revision 1.png (mean rank of every method, flat CV vs nested CV)
# ==============================================================================

library(openxlsx)
library(ggplot2)

DIR   <- "results/nested CV - revision 1"
ranks <- read.csv(file.path(DIR, "nested CV mean ranks - revision 1.csv"))
vs    <- read.csv(file.path(DIR, "nested CV nested vs flat - revision 1.csv"))
dir.create("results/Figures", showWarnings = FALSE)

label <- c(RF = "RF", NaiveBayes = "NB", CART = "DT", SVMRadial = "SVM", KNN = "KNN", LogReg = "LogReg",
           SMOTE_RF = "SMOTE + RF", RUS_RF = "RUS + RF", SMOTE_SVM = "SMOTE + SVM", NNET = "NN", BalancedNNET = "BalNN",
           SMOTENNET = "SMOTE + NN", FocalNNET = "FocalNN", FocalSDNNET = "FocalSDNN", LDAMNNET = "LDAMNN", MMNNET = "MMNN",
           PNN = "PNN", SkewPNN = "SkewPNN", CoverNET = "CoverNET")
mlab  <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", ROCAUC = "AUC")

# ------------------------------------------------------------------------------
# Table 1 and Figure 1: nested vs flat CV, tuned neural methods
# ------------------------------------------------------------------------------
t1 <- vs[vs$Reference == "flat CV", ]
t1$HolmP   <- ave(t1$WilcoxonP, t1$Metric, FUN = function(p) p.adjust(p, "holm"))
t1$Metric  <- factor(mlab[t1$Metric], levels = mlab)
t1$Method  <- label[t1$Method]
t1 <- t1[order(t1$Metric), ]

table1 <- data.frame(Metric = t1$Metric, Method = t1$Method,
                     "Nested lower" = t1$NestedLower, "Equal" = t1$Tie, "Nested higher" = t1$NestedHigher,
                     "Holm p" = ifelse(t1$HolmP < 0.001, "<0.001", formatC(t1$HolmP, 3, format = "f")),
                     check.names = FALSE)

long <- rbind(data.frame(t1[, c("Metric", "Method")], Result = "Nested lower",  n = t1$NestedLower),
              data.frame(t1[, c("Metric", "Method")], Result = "Equal",         n = t1$Tie),
              data.frame(t1[, c("Metric", "Method")], Result = "Nested higher", n = t1$NestedHigher))
long$Result <- factor(long$Result, levels = c("Nested lower", "Equal", "Nested higher"))
ord <- names(sort(tapply(t1$NestedLower, t1$Method, sum)))                       # most affected method on top
long$Method <- factor(long$Method, levels = ord)

plt1 <- ggplot(long, aes(x = Method, y = n, fill = Result)) +
  geom_col(width = 0.7, linewidth = 0.25, color = "grey30") +
  scale_fill_manual(name = NULL, values = c("Nested lower" = "darkorange", "Equal" = "grey80", "Nested higher" = "darkgreen")) +
  coord_flip() +
  facet_wrap(~ Metric, nrow = 1) +
  labs(x = NULL, y = "Number of datasets") +
  theme_bw() +
  theme(legend.position = "bottom", strip.background = element_rect(fill = NA, color = NA))
ggsave("results/Figures/nested vs flat counts - revision 1.png", plot = plt1, width = 12, height = 4.5, dpi = 300)

# ------------------------------------------------------------------------------
# Table 2 and Figure 2: mean rank of the 19 methods of the paper, flat CV vs nested CV
# ------------------------------------------------------------------------------
fl <- ranks[ranks$Scenario == "paper flat",           c("Metric", "Method", "MeanRank")]
ns <- ranks[ranks$Scenario == "paper methods nested", c("Metric", "Method", "MeanRank")]
rk <- merge(fl, ns, by = c("Metric", "Method"))
names(rk)[3:4] <- c("flat", "nested")
rk$PositionFlat   <- ave(rk$flat,   rk$Metric, FUN = function(v) rank(v, ties.method = "min"))
rk$PositionNested <- ave(rk$nested, rk$Metric, FUN = function(v) rank(v, ties.method = "min"))
rk$Metric <- factor(mlab[rk$Metric], levels = mlab)
rk$Method <- label[rk$Method]
rk <- rk[order(rk$Metric, rk$nested), ]

table2 <- data.frame(Metric = rk$Metric, Method = rk$Method,
                     "Mean rank flat CV" = round(rk$flat, 2), "Position flat CV" = rk$PositionFlat,
                     "Mean rank nested CV" = round(rk$nested, 2), "Position nested CV" = rk$PositionNested,
                     check.names = FALSE)

rk$CoverNET <- rk$Method == "CoverNET"
rk$Method   <- factor(rk$Method, levels = rev(names(sort(tapply(rk$nested, rk$Method, mean)))))
plt2 <- ggplot(rk, aes(y = Method)) +
  geom_segment(aes(x = flat, xend = nested, yend = Method), color = "grey60") +
  geom_point(aes(x = flat,   shape = "Flat CV",   color = CoverNET), size = 2.2, stroke = 0.8) +
  geom_point(aes(x = nested, shape = "Nested CV", color = CoverNET), size = 2.2) +
  scale_color_manual(values = c("FALSE" = "black", "TRUE" = "red"), guide = "none") +
  scale_shape_manual(name = NULL, values = c("Flat CV" = 1, "Nested CV" = 16)) +
  facet_wrap(~ Metric, nrow = 1) +
  labs(x = "Mean rank (lower is better)", y = NULL) +
  theme_bw() +
  theme(legend.position = "bottom", strip.background = element_rect(fill = NA, color = NA))
ggsave("results/Figures/mean rank flat vs nested - revision 1.png", plot = plt2, width = 12, height = 5, dpi = 300)

write.xlsx(list("Nested vs flat" = table1, "Mean ranks" = table2),
           file.path(DIR, "nested CV appendix tables - revision 1.xlsx"))

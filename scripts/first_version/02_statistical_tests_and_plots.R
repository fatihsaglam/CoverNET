library(openxlsx)
library(ggplot2)
library(reshape2)
library(dplyr)
library(bbtcomp)
library(ggpubr)
library(ggsignif)
library(bayesplot)

results_pvalues <- read.xlsx(xlsxFile = "results/manuscript tables.xlsm",
                     sheet = "Statistical tests")
results_combined <- read.xlsx(xlsxFile = "results/manuscript tables.xlsm",
                           sheet = "Combined")

results_ranks <- results_combined[, c(1:2, 3:21)]
results_perfs <- results_combined[, c(1:2, 22:40)]

results_ranks
results_ranks_melted <- reshape2:::melt.data.frame(
  results_ranks,
  measure.vars = 3:21, value.name = "Rank", variable.name = "Classifier"
)

levels(results_ranks_melted$Classifier) <- sub(
  "\\+.", "+ ",
  sub("\\.", " ", sub("_rank", "", levels(results_ranks_melted$Classifier)))
)

plt_box_rank <- ggplot(results_ranks_melted) +
  stat_boxplot(mapping = aes(
    x = Classifier,
    y = Rank
  ), geom = "errorbar", width = 0.35) +
  geom_boxplot(mapping = aes(
    x = Classifier,
    y = Rank
  ), fill = "Orange") +
  facet_wrap(.~Metric, nrow = 2, ncol = 2) +
  scale_x_discrete(guide = guide_axis(angle = 45)) +
  theme_bw() +
  theme(strip.background = element_rect(fill = NA, color = NA))

results_perfs_melted <- reshape2:::melt.data.frame(
  results_perfs,
  measure.vars = 3:21, value.name = "Rank", variable.name = "Classifier"
)

levels(results_perfs_melted$Classifier) <- levels(results_ranks_melted$Classifier)

plt_box_perf <- ggplot(results_perfs_melted) +
  stat_boxplot(mapping = aes(
    x = Classifier,
    y = Rank
  ), geom = "errorbar", width = 0.35) +
  geom_boxplot(mapping = aes(
    x = Classifier,
    y = Rank
  ), fill = "Orange") +
  facet_wrap(.~Metric, nrow = 2, ncol = 2, scale = "free_y") +
  scale_x_discrete(guide = guide_axis(angle = 45)) +
  theme_bw() +
  theme(strip.background = element_rect(fill = NA, color = NA))

### Bradley T model

results_SDs <- read.xlsx(xlsxFile = "results/results combined.xlsx", sheet = "SDs")
results_SDs <- results_SDs[results_SDs$Dataset %in% unique(results_combined$Dataset) &
                             results_SDs$Metric %in% c("IAM", "MCC", "P4", "ROCAUC"),]
results_SDs$Metric[results_SDs$Metric == "ROCAUC"] <- "AUC"

results_SDs <- results_SDs[
  match(
    paste(results_perfs$Dataset, results_perfs$Metric),
    paste(results_SDs$Dataset,   results_SDs$Metric)
  ),
]

m_bbtcomp_IAM <- bbtcomp(tabx = results_perfs[results_perfs$Metric == "IAM",c(3:21)], 
                         tabsd = results_SDs[results_SDs$Metric == "IAM",c(3:21)])
m_bbtcomp_MCC <- bbtcomp(tabx = results_perfs[results_perfs$Metric == "MCC",c(3:21)],
                         tabsd = results_SDs[results_SDs$Metric == "MCC",c(3:21)])
m_bbtcomp_P4 <- bbtcomp(tabx = results_perfs[results_perfs$Metric == "P4",c(3:21)],
                        tabsd = results_SDs[results_SDs$Metric == "P4",c(3:21)])
m_bbtcomp_AUC <- bbtcomp(tabx = results_perfs[results_perfs$Metric == "AUC",c(3:21)],
                         tabsd = results_SDs[results_SDs$Metric == "AUC",c(3:21)])

make_plotData <- \(m, value.name) {
  m_pwin <- bbtcomp:::get_pwin(m)
  m_pwin_CoverNet <- as.data.frame(m_pwin) |> select(contains("CoverNet"))
  colnames(m_pwin_CoverNet) <- sub("\\+.", "+ ", sub("\\.", " ", colnames(m_pwin_CoverNet)))
  m_pwin_CoverNet_melted <- reshape2:::melt.data.frame(m_pwin_CoverNet, value.name = "Performance", variable.name = "Classifier")
  data.frame(Metric = value.name, m_pwin_CoverNet_melted)
}

m_bbtcomp_plotData_IAM <- make_plotData(m = m_bbtcomp_IAM, value.name = "IAM")
m_bbtcomp_plotData_MCC <- make_plotData(m = m_bbtcomp_MCC, value.name = "MCC")
m_bbtcomp_plotData_P4 <- make_plotData(m = m_bbtcomp_P4, value.name = "P4")
m_bbtcomp_plotData_AUC <- make_plotData(m = m_bbtcomp_AUC, value.name = "AUC")

m_bbtcomp_plotData_combined <- rbind(
  m_bbtcomp_plotData_IAM,
  m_bbtcomp_plotData_MCC,
  m_bbtcomp_plotData_P4,
  m_bbtcomp_plotData_AUC
)
m_bbtcomp_plotData_combined$Metric <- as.factor(m_bbtcomp_plotData_combined$Metric)
m_bbtcomp_plotData_combined$Classifier <- as.factor(m_bbtcomp_plotData_combined$Classifier)

plot_bbtcomp <- m_bbtcomp_plotData_combined |>
  group_by(Classifier, Metric) |>
  summarise(
    xmin    = min(Performance),
    xlower  = quantile(Performance, 0.025),
    xmiddle = median(Performance),
    xupper  = quantile(Performance, 0.975),
    xmax    = max(Performance),
    .groups = "drop"
  ) |>
  ggplot(aes(y = Classifier)) +
  geom_vline(xintercept = 0.5, color = "black", linetype = 2) +
  geom_vline(xintercept = c(0.45, 0.55), colour = "red", linetype = 2) +
  geom_boxplot(
    aes(xmin = xmin, xlower = xlower, xmiddle = xmiddle, xupper = xupper, xmax = xmax),
    stat = "identity", fill = "orange"
  ) +
  facet_wrap(Metric ~ .) +
  scale_x_continuous(name = "Posterior") +
  theme_bw() +
  theme(
    axis.text.x       = element_text(angle = 0, hjust = 0.5, size = 12, color = "black"),
    axis.text.y       = element_text(size = 12, color = "black"),
    strip.background  = element_rect(fill = NA, colour = NA),
    strip.text = element_text(size = 12, color = "black"),
    text = element_text(size = 12, color = "black")
  )

ggsave(filename = "results/Figures/Bayesian comparison boxplot.png", plot = plot_bbtcomp, 
       width = 12, height = 7, dpi = 300)


wintable_IAM <- data.frame(Metric = "IAM", table_wintable(m_bbtcomp_IAM$wintable, which = "both"))
wintable_MCC <- data.frame(Metric = "MCC", table_wintable(m_bbtcomp_MCC$wintable, which = "both"))
wintable_P4 <- data.frame(Metric = "P4", table_wintable(m_bbtcomp_P4$wintable, which = "both"))
wintable_AUC <- data.frame(Metric = "AUC", table_wintable(m_bbtcomp_AUC$wintable, which = "both"))

wintable <- rbind(
  wintable_IAM,
  wintable_MCC,
  wintable_P4,
  wintable_AUC
)

probTable_IAM <- data.frame(Metric = "IAM", table_pwin(m_bbtcomp_IAM, control = "CoverNet"))
probTable_MCC <- data.frame(Metric = "MCC", table_pwin(m_bbtcomp_MCC, control = "CoverNet"))
probTable_P4 <- data.frame(Metric = "P4", table_pwin(m_bbtcomp_P4, control = "CoverNet"))
probTable_AUC <- data.frame(Metric = "AUC", table_pwin(m_bbtcomp_AUC, control = "CoverNet"))

probTable <- rbind(
  probTable_IAM,
  probTable_MCC,
  probTable_P4,
  probTable_AUC
)

write.xlsx(x = list(
  probTable = probTable,
  winTable = wintable[wintable$alg2 %in% "CoverNet",]
), file = "results/Bradley Terry model results.xlsx")



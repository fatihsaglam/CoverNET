library(openxlsx)
library(ggplot2)
library(dplyr)

results <- read.xlsx(xlsxFile = "results/manuscript tables.xlsm", sheet = "Combined")

results_melted <- reshape2:::melt.data.frame(
  data         = results[, c(1:21, 41:45)],
  measure.vars = colnames(results)[3:21],
  variable.name = "Classifiers",
  value.name    = "Rank"
)

levels(results_melted$Classifiers) <- sub(
  "\\+.", "+ ",
  sub("\\.", " ", sub("_rank", "", levels(results_melted$Classifiers)))
)

# ==============================================================================
# CREATE BIN GROUPS
# ==============================================================================

# Helper: bin into 4 quartile groups with readable labels
bin_quartiles <- function(x, name = "x") {
  qs <- quantile(x, probs = c(0, 0.25, 0.5, 0.75, 1.0), na.rm = TRUE)
  labels <- sprintf("[%g, %g]",
                    if (name == "p/n") round(qs[-length(qs)], 3) else round(qs[-length(qs)]),
                    if (name == "p/n") round(qs[-1],          3) else round(qs[-1]))
  cut(x,
      breaks         = qs,
      labels         = labels,
      include.lowest = TRUE)
}

results_melted$n_bin   <- bin_quartiles(results_melted$n,             "n")
results_melted$p_bin   <- bin_quartiles(results_melted$p,             "p")
results_melted$IR_bin  <- bin_quartiles(results_melted$IR,            "IR")
results_melted$pn_bin  <- bin_quartiles(results_melted$dimensionality,"p/n")

# ==============================================================================
# PLOTTING HELPER
# ==============================================================================

make_box_plot <- function(data, group_var, group_label) {
  ggplot(data,
         aes_string(x    = group_var,
                    y    = "Rank",
                    fill = "Classifiers")) +
    stat_boxplot(geom = "errorbar",
                 width = 0.5,
                 linewidth = 0.2,
                 position = position_dodge(width = 0.9)) +
    geom_boxplot(outlier.size = 0.5, linewidth = 0.2, width = 0.9, 
                 position = position_dodge(width = 0.9)) +
    facet_wrap(. ~ Metric, nrow = 2, ncol = 2) +
    geom_vline(xintercept = seq(0.5, 4.5, 1), 
               linetype  = 2,
               linewidth = 0.3,
               color     = "gray60") +
    scale_fill_manual(values = rev(ggsci::pal_d3("category20")(19))) +
    scale_x_discrete(expand = c(0.02, 0.02)) +
    labs(x = group_label) +
    theme_bw() +
    guides(fill = guide_legend(nrow = 2)) +
    theme(
      legend.position   = "top",
      legend.key.height = unit(0.3, "cm"),
      axis.text.x       = element_text(angle = 0, hjust = 0.5, size = 12, color = "black"),
      axis.text.y       = element_text(size = 12, color = "black"),
      strip.background  = element_rect(fill = NA, colour = NA),
      strip.text = element_text(size = 12, color = "black"),
      text = element_text(size = 12, color = "black")
    )
}

# ==============================================================================
# CREATE PLOTS
# ==============================================================================

plot_n  <- make_box_plot(results_melted, "n_bin",  "Sample size (n)")
plot_p  <- make_box_plot(results_melted, "p_bin",  "Number of features (p)")
plot_IR <- make_box_plot(results_melted, "IR_bin", "Imbalance ratio (IR)")
plot_pn <- make_box_plot(results_melted, "pn_bin", "p/n")

# ==============================================================================
# SAVE
# ==============================================================================

ggsave("results/Figures/rank vs n box.png",  plot = plot_n,
       height = 4, width = 12, dpi = 300)
ggsave("results/Figures/rank vs p box.png",  plot = plot_p,
       height = 4, width = 12, dpi = 300)
ggsave("results/Figures/rank vs IR box.png", plot = plot_IR,
       height = 4, width = 12, dpi = 300)
ggsave("results/Figures/rank vs pn box.png", plot = plot_pn,
       height = 4, width = 12, dpi = 300)

results_all <- read.xlsx(xlsxFile = "results/results combined.xlsx", "Means")
results_all_filtered <- 
  results_all[results_all$Dataset %in% results$Dataset &
                results_all$Metric %in% c("IAM", "MCC", "P4", "ROCAUC", "Runtime"),]


# Filter to Runtime only, melt to long format
runtime_data <- results_all_filtered[results_all_filtered$Metric == "Runtime", ]

runtime_melted <- reshape2:::melt.data.frame(
  data         = runtime_data,
  measure.vars = colnames(runtime_data)[3:21],
  variable.name = "Classifiers",
  value.name    = "Runtime"
)

# Clean classifier names same as before
levels(runtime_melted$Classifiers) <- sub(
  "\\+.", "+ ",
  sub("\\.", " ", sub("_rank", "", levels(results_melted$Classifiers)))
)

ds_chars <- unique(results[, c("Dataset", "n", "p", "IR", "dimensionality")])
runtime_melted <- merge(runtime_melted, ds_chars, by = "Dataset")

runtime_melted$n_bin  <- bin_quartiles(runtime_melted$n,             "n")
runtime_melted$p_bin  <- bin_quartiles(runtime_melted$p,             "p")
runtime_melted$IR_bin <- bin_quartiles(runtime_melted$IR,            "IR")
runtime_melted$pn_bin <- bin_quartiles(runtime_melted$dimensionality,"p/n")

# Floor near-zero runtimes for log scale
runtime_melted$Runtime <- pmax(runtime_melted$Runtime, 0.001)

# ==============================================================================
# RESHAPE FOR FACETING
# ==============================================================================

library(tidyr)

# Convert wide bins to long format
runtime_long <- runtime_melted %>%
  pivot_longer(
    cols      = c(n_bin, p_bin, IR_bin, pn_bin),
    names_to  = "Characteristic",
    values_to = "Group"
  )

# Make Characteristic a factor with readable labels and desired order
runtime_long$Characteristic <- factor(
  runtime_long$Characteristic,
  levels = c("n_bin", "p_bin", "IR_bin", "pn_bin"),
  labels = c("Sample size (n)",
             "Number of features (p)",
             "Imbalance ratio (IR)",
             "p/n")
)

# ==============================================================================
# SINGLE FACETED PLOT
# ==============================================================================

plot_runtime_combined <- ggplot(runtime_long,
                                aes(x = Group, y = Runtime, fill = Classifiers)) +
  stat_boxplot(geom = "errorbar",
               width = 0.5,
               linewidth = 0.2,
               position = position_dodge(width = 0.9)) +
  geom_boxplot(outlier.size = 0.5, linewidth = 0.2,
               width = 0.9, position = position_dodge(width = 0.9)) +
  facet_wrap(. ~ Characteristic, scales = "free_x", nrow = 2, ncol = 2) +
  scale_x_discrete(expand = c(0.02, 0.02)) +
  scale_y_log10(name = "Runtime (seconds, log scale)") +
  scale_fill_manual(name = "", values = rev(ggsci::pal_d3("category20")(19))) +
  geom_vline(xintercept = seq(0.5, 4.5, 1), 
             linetype  = 2,
             linewidth = 0.3,
             color     = "gray60") +
  labs(x = NULL) +
  theme_bw() +
  guides(fill = guide_legend(nrow = 2)) +
  theme(
    legend.position   = "top",
    legend.key.height = unit(0.3, "cm"),
    axis.text.x       = element_text(angle = 0, hjust = 0.5, size = 12, color = "black"),
    axis.text.y       = element_text(size = 12, color = "black"),
    strip.background  = element_rect(fill = NA, colour = NA),
    strip.text = element_text(size = 12, color = "black"),
    text = element_text(size = 12, color = "black")
  )

ggsave("results/Figures/runtime combined.png",
       plot   = plot_runtime_combined,
       height = 7, width = 11, dpi = 300)

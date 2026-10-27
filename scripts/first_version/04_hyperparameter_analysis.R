library(openxlsx)
library(ggplot2)
library(dplyr)
library(tidyr)

results_combined <- read.xlsx(xlsxFile = "results/manuscript tables.xlsm",
                              sheet = "Combined")
bestParams_all <- read.xlsx(xlsxFile = "results/results combined.xlsx", sheet = "BestParams")

bestParams <- bestParams_all[bestParams_all$Dataset %in% unique(results_combined$Dataset),]

bestParams_CCCDNN <- bestParams[bestParams$Method == "CoverNET", ]

colnames(bestParams_CCCDNN)
bestParams_CCCDNN$BestParams


# ==============================================================================
# 1. PARSE BESTPARAMS STRINGS INTO COLUMNS
# ==============================================================================

parse_params <- function(param_str) {
  pairs <- strsplit(param_str, ", ")[[1]]
  kv <- strsplit(pairs, "=")
  values <- sapply(kv, function(x) x[2])
  names(values) <- sapply(kv, function(x) x[1])
  as.list(values)
}

# Apply to all rows and bind into a data frame
parsed_list <- lapply(bestParams_CCCDNN$BestParams, parse_params)
params_df   <- do.call(rbind, lapply(parsed_list, as.data.frame, stringsAsFactors = FALSE))

# Combine with dataset info
hyper_df <- cbind(
  Dataset = bestParams_CCCDNN$Dataset,
  params_df
)

# Convert to numeric where appropriate
hyper_df$hidden_dim <- as.integer(hyper_df$hidden_dim)
hyper_df$n_hidden   <- as.integer(hyper_df$n_hidden)
hyper_df$lambda1    <- as.numeric(hyper_df$lambda1)
hyper_df$lambda2    <- as.numeric(hyper_df$lambda2)
hyper_df$lambda3    <- as.numeric(hyper_df$lambda3)

# ==============================================================================
# 2. VISUALIZATIONS
# ==============================================================================

# --- Discrete hyperparameters: hidden_dim and n_hidden as bar charts ---

p_hidden_dim <- ggplot(hyper_df, aes(x = factor(hidden_dim))) +
  geom_bar(fill = "#1f77b4", alpha = 0.8) +
  labs(x = "Hidden dimension", y = "Number of datasets") +
  theme_bw() +
  theme(strip.background = element_rect(fill = NA, colour = NA))

p_n_hidden <- ggplot(hyper_df, aes(x = factor(n_hidden))) +
  geom_bar(fill = "#ff7f0e", alpha = 0.8) +
  labs(x = "Number of hidden layers", y = "Number of datasets") +
  theme_bw() +
  theme(strip.background = element_rect(fill = NA, colour = NA))

# --- Continuous hyperparameters: lambdas as histograms ---

lambda_long <- hyper_df %>%
  pivot_longer(cols = c(lambda1, lambda2, lambda3),
               names_to  = "Parameter",
               values_to = "Value")

# Pretty labels
lambda_long$Parameter <- factor(lambda_long$Parameter,
                                levels = c("lambda1", "lambda2", "lambda3"),
                                labels = c(expression(lambda[1]),
                                           expression(lambda[2]),
                                           expression(lambda[3])))

p_lambdas <- ggplot(lambda_long, aes(x = Value)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 20,
                 fill = "#2ca02c",
                 alpha = 0.6,
                 color = "white") +
  geom_density(color = "#1a5e1a",
               linewidth = 0.8,
               fill = NA) +
  facet_wrap(. ~ Parameter, scales = "free", nrow = 1,
             labeller = label_parsed) +
  labs(x = "Selected value", y = "Density") +
  theme_bw() +
  theme(strip.background = element_rect(fill = NA, colour = NA))

# ==============================================================================
# 3. COMBINE WITH PATCHWORK
# ==============================================================================

library(patchwork)

combined_hyper <- (p_hidden_dim | p_n_hidden) / p_lambdas 

ggsave("results/Figures/CoverNet hyperparameters.png",
       plot = combined_hyper, height = 6*0.7, width = 12*0.7, dpi = 300)



dataset_info_selected <- dataset_info[dataset_info$dataset_name %in% unique(bestParams_CCCDNN$Dataset),]

hyper_with_chars <- cbind(hyper_df, dataset_info_selected[, -1])

library(tidyr)
library(dplyr)

# Pivot lambdas to long format
hyper_long <- hyper_with_chars %>%
  pivot_longer(cols = c(lambda1, lambda2, lambda3),
               names_to  = "Lambda",
               values_to = "Value")

# Plot lambdas vs all characteristics
char_long <- hyper_long %>%
  pivot_longer(cols = c(n, p, IR, dimensionality),
               names_to  = "Characteristic",
               values_to = "CharValue")

plt_lambdas <- ggplot(char_long, aes(x = CharValue, y = Value)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE) +
  stat_cor(method = "spearman",
           size = 3,
           label.x.npc = 0.05,
           label.y.npc = 0.95,
           geom = "label",
           fill = "white",
           alpha = 0.8,
           label.padding = unit(0.15, "lines")) +
  facet_grid(Lambda ~ Characteristic, scales = "free",
             labeller = labeller(
               Lambda = c(lambda1 = "λ₁",
                          lambda2 = "λ₂",
                          lambda3 = "λ₃"),
               Characteristic = c(IR             = "Imbalance ratio",
                                  dimensionality = "p/n",
                                  n              = "Sample size",
                                  p              = "Features")
             )) +
  scale_x_log10(name = "") +
  theme_bw()

ggsave("results/Figures/lambda hyperparameters.png",
       plot = plt_lambdas, height = 6*0.8, width = 12*0.8, dpi = 300)


arch_long <- hyper_with_chars %>%
  pivot_longer(cols = c(hidden_dim, n_hidden),
               names_to  = "Parameter",
               values_to = "Value") %>%
  pivot_longer(cols = c(n, p, IR, dimensionality),
               names_to  = "Characteristic",
               values_to = "CharValue")

library(ggpubr)

library(patchwork)

arch_long_hd <- arch_long[arch_long$Parameter == "hidden_dim", ]
arch_long_hd <- arch_long_hd[arch_long_hd$Value != 64, ]  # drop rare value

arch_long_nh <- arch_long[arch_long$Parameter == "n_hidden", ]
arch_long_nh <- arch_long_nh[arch_long_nh$Value != 1, ]   # drop rare value, keep 2 groups

# --- Hidden dim: 3 groups → Kruskal-Wallis ---
p_hd <- ggplot(arch_long_hd, aes(x = factor(Value), y = CharValue)) +
  geom_boxplot(outlier.size = 0.8, fill = "#1f77b4", alpha = 0.6) +
  stat_compare_means(method = "kruskal.test", size = 3,
                     label.y.npc = 0.92) +
  facet_wrap(. ~ Characteristic, scales = "free_y", nrow = 1,
             labeller = labeller(Characteristic = c(
               IR             = "Imbalance ratio",
               dimensionality = "p/n",
               n              = "Sample size",
               p              = "Features"))) +
  scale_y_log10() +
  labs(x = "Hidden dim", y = "Characteristic value") +
  theme_bw() +
  theme(strip.background = element_rect(fill = NA, colour = NA))

# --- N hidden: 2 groups → Wilcoxon ---
p_nh <- ggplot(arch_long_nh, aes(x = factor(Value), y = CharValue)) +
  geom_boxplot(outlier.size = 0.8, fill = "#ff7f0e", alpha = 0.6) +
  stat_compare_means(method = "wilcox.test", size = 3,
                     label.y.npc = 0.92) +
  facet_wrap(. ~ Characteristic, scales = "free_y", nrow = 1,
             labeller = labeller(Characteristic = c(
               IR             = "Imbalance ratio",
               dimensionality = "p/n",
               n              = "Sample size",
               p              = "Features"))) +
  scale_y_log10() +
  labs(x = "Hidden layers", y = "Characteristic value") +
  theme_bw() +
  theme(strip.background = element_rect(fill = NA, colour = NA))

# --- Combine ---
combined <- p_hd / p_nh
ggsave("results/Figures/arch hyperparameters.png",
       plot = combined, height = 4.5, width = 9, dpi = 300)

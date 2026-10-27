source("global.R")

data <- datasets_binary$banana
p <- ncol(data) - 1
n <- nrow(data)

x <- data[,1:p]
y <- data[,p+1]

res <- 150
x1_grid <- seq(min(x[,1]), max(x[,1]), length.out = res)
x2_grid <- seq(min(x[,2]), max(x[,2]), length.out = res)
grid <- expand.grid(x1_grid, x2_grid)

set.seed(1)
torch$manual_seed(1)

m_LDAMNNET <- classifier_LDAMNNET$new()
m_LDAMNNET$fit(
  x = x, 
  y = y, 
  hidden_dim = 32L, 
  n_hidden = 2L, 
  activation = "relu",
  epochs = 5L, 
  batch_size = 64L, 
  lr = 0.001, 
  patience = 10L
)

m_preds_grid <- m_LDAMNNET$predict(grid)

probs_grid <- m_preds_grid$prob

probs_grid[probs_grid > 1 - 1e-4] <- 1 - 1e-4
probs_grid[probs_grid < 1e-4] <- 1e-4

epochs_seq <- as.integer(c(1:5, 25, 50))
plts_steps <- list()
for (i_epoch in 1:length(epochs_seq)) {
  
  set.seed(1)
  torch$manual_seed(1)
  m_CoverNet <- classifier_CCCDNNET$new()
  m_CoverNet$fit(
    x = x, 
    y = y, 
    hidden_dim = 32L, 
    n_hidden = 2L, 
    activation = "relu", 
    epochs = epochs_seq[i_epoch], 
    batch_size = 64L,
    lr = 0.001,
    k_pos = NULL, 
    k_neg = NULL, 
    lambda1 = 1, 
    lambda2 = 1, 
    lambda3 = 1, 
    patience = 10
  )
  m_preds_grid <- m_CoverNet$predict(grid)
  
  probs_grid <- m_preds_grid$prob
  
  probs_grid[probs_grid > 1 - 1e-4] <- 1 - 1e-4
  probs_grid[probs_grid < 1e-4] <- 1e-4
  
  plts_steps[[i_epoch]] <- local({
    i_epoch <- i_epoch
    probs_grid <- probs_grid
    ggplot() +
      geom_contour_filled(mapping = aes(
        x = grid[,1],
        y = grid[,2],
        z = probs_grid[,1]
      ), breaks = seq(0, 1, length = 11)) +
      geom_contour(mapping = aes(
        x = grid[,1],
        y = grid[,2],
        z = probs_grid[,1]
      ), breaks = 0.5, color = "black") +
      guides(fill = guide_legend(nrow = 2, title = NULL, position = "top", direction = "vertical", label.position = "bottom")) +
      ggnewscale::new_scale_fill() +
      guides(fill = guide_legend(nrow = 1, title = NULL, position = "bottom")) +
      labs(title = paste0("Epoch ", epochs_seq[i_epoch])) +
      geom_point(mapping = aes(
        x = x[,1],
        y = x[,2],
        fill = y
      ), alpha = 0.5, shape = 21, stroke = 0.25, size = 2.5) +
      ggsci::scale_fill_d3(name = "") +
      scale_x_continuous(name = NULL, expand = c(0,0), breaks = NULL) +
      scale_y_continuous(name = NULL, expand = c(0,0), breaks = NULL) +
      theme_bw() +
      theme(legend.text = element_text(size = 8),
            legend.key.size = unit(0.5, "cm"))
  })
}


# local({
#   i_epoch <- i_epoch
#   probs_grid <- probs_grid
#   ggplot() +
#     geom_contour_filled(mapping = aes(
#       x = grid[,1],
#       y = grid[,2],
#       z = probs_grid[,1]
#     ), breaks = seq(0, 1, length = 11)) +
#     geom_contour(mapping = aes(
#       x = grid[,1],
#       y = grid[,2],
#       z = probs_grid[,1]
#     ), breaks = 0.5, color = "black") +
#     guides(fill = guide_legend(
#       nrow = 2, 
#       title = NULL,
#       position = "top", 
#       direction = "vertical", 
#       label.position = "bottom")) +
#     ggnewscale::new_scale_fill() +
#     guides(fill = guide_legend(nrow = 1, title = NULL, position = "bottom")) +
#     labs(title = paste0("Epoch ", epochs_seq[i_epoch])) +
#     geom_point(mapping = aes(
#       x = x[,1],
#       y = x[,2],
#       fill = y
#     ), alpha = 0.5, shape = 21, stroke = 0.25, size = 2.5) +
#     ggsci::scale_fill_d3(name = "") +
#     scale_x_continuous(name = NULL, expand = c(0,0), breaks = NULL) +
#     scale_y_continuous(name = NULL, expand = c(0,0), breaks = NULL) +
#     theme_bw() +
#     theme(
#       legend.text = element_text(size = 8),
#       legend.key.size = unit(0.5, "cm")
#     )
# })


library(cowplot)
# plt_legend <- get_plot_component(plts_steps[[1]], "guide-box", FALSE)
# get_legend
plt_title <- get_title(plts_steps[[1]])
plt_title$children[[1]]$label <- "CoverNet"
# for (i in 1:length(plts_steps)) {
#   plts_steps[[i]] <- plts_steps[[i]] + theme(legend.position = "none")
# }

plt_CoverNet_epochContours <- plot_grid(
  plt_title,
  plot_grid(
    plotlist = plts_steps, nrow = 1, ncol = length(plts_steps)
  ),
  nrow = 2, ncol = 1, rel_heights = c(0.1, 0.9)
)

# plt_CoverNet_epochContours <- ggpubr::ggarrange(
#   plotlist = plts_steps,
#   nrow = 1, ncol = length(plts_steps),
#   common.legend = TRUE, legend = "top"
# )

ggsave(filename = "results/Figures/plt_CoverNet_epochContours.png", 
       plot = plt_CoverNet_epochContours,
       width = 20, height = 5, dpi = 300)

epochs_seq <- as.integer(c(1:5, 25, 50))
plts_steps <- list()

for (i_epoch in 1:length(epochs_seq)) {
  
  set.seed(1)
  torch$manual_seed(1)
  m_FocalNNET <- classifier_FocalNNET$new()
  m_FocalNNET$fit(
    x = x, 
    y = y, 
    hidden_dim = 32L, 
    n_hidden = 2L, 
    activation = "relu", 
    epochs = epochs_seq[i_epoch], 
    batch_size = 64L,
    lr = 0.001,
    gamma = 1, 
    patience = 10
  )
  m_preds_grid <- m_FocalNNET$predict(grid)
  
  probs_grid <- m_preds_grid$prob
  
  probs_grid[probs_grid > 1 - 1e-4] <- 1 - 1e-4
  probs_grid[probs_grid < 1e-4] <- 1e-4
  
  plts_steps[[i_epoch]] <- local({
    i_epoch <- i_epoch
    probs_grid <- probs_grid
    ggplot() +
      geom_contour_filled(mapping = aes(
        x = grid[,1],
        y = grid[,2],
        z = probs_grid[,1]
      ), breaks = seq(0, 1, length = 11)) +
      geom_contour(mapping = aes(
        x = grid[,1],
        y = grid[,2],
        z = probs_grid[,1]
      ), breaks = 0.5, color = "black") +
      guides(fill = guide_legend(nrow = 2, title = NULL, position = "top", direction = "vertical", label.position = "bottom")) +
      ggnewscale::new_scale_fill() +
      guides(fill = guide_legend(nrow = 1, title = NULL, position = "bottom")) +
      labs(title = paste0("Epoch ", epochs_seq[i_epoch])) +
      geom_point(mapping = aes(
        x = x[,1],
        y = x[,2],
        fill = y
      ), alpha = 0.5, shape = 21, stroke = 0.25, size = 2.5) +
      ggsci::scale_fill_d3(name = "") +
      scale_x_continuous(name = NULL, expand = c(0,0), breaks = NULL) +
      scale_y_continuous(name = NULL, expand = c(0,0), breaks = NULL) +
      theme_bw() +
      theme(legend.text = element_text(size = 8),
            legend.key.size = unit(0.5, "cm"))
  })
}

library(cowplot)
# plt_legend <- get_plot_component(plts_steps[[1]], "guide-box", FALSE)
# get_legend
plt_title <- get_title(plts_steps[[1]])
plt_title$children[[1]]$label <- "FocalNN"
# for (i in 1:length(plts_steps)) {
#   plts_steps[[i]] <- plts_steps[[i]] + theme(legend.position = "none")
# }

plt_FocalNNET_epochContours <- plot_grid(
  plt_title,
  # plt_legend,
  plot_grid(
    plotlist = plts_steps, nrow = 1, ncol = length(plts_steps)
  ),
  nrow = 2, ncol = 1, rel_heights = c(0.05, 0.9)
)


# plt_FocalNNET_epochContours <- ggpubr::ggarrange(
#   plotlist = plts_steps,
#   nrow = 1, ncol = length(plts_steps),
#   common.legend = TRUE, legend = "top"
# )

ggsave(filename = "results/Figures/plt_FocalNNET_epochContours.png", 
       plot = plt_FocalNNET_epochContours,
       width = 20, height = 5, dpi = 300)

library(cowplot)

plot_epochCountersComparison <- plot_grid(
  plt_CoverNet_epochContours,
  plt_FocalNNET_epochContours,
  nrow = 2, ncol = 1
)
ggsave(filename = "results/Figures/plot_epochCountersComparison.png", 
       plot = plot_epochCountersComparison,
       width = 20, height = 10, dpi = 300)


### learning curves 
data <- datasets_binary$banana
p <- ncol(data) - 1
n <- nrow(data)

x <- data[,1:p]
y <- data[,p+1]


names(methods_list)


set.seed(1)
torch$manual_seed(1)
methods_NNs <- c("NNET", "BalancedNNET",
  "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "CCCDNNET")
epochs_seq <- 1:50

i_train <- createDataPartition(y = y, times = 1, p = 0.75, list = FALSE)
x_train <- x[i_train,]
y_train <- y[i_train]

x_test <- x[-i_train,]
y_test <- y[-i_train]

results <- list()
for (method_name in methods_NNs) {
  results[[method_name]] <- list()
  for (epoch in epochs_seq) {
    m <- methods_list[[method_name]]$classifier_class$new()
    m$fit(x = x_train, y = y_train, epochs = epoch, patience = 1234L)
    m_pred_train <- m$predict(x = x_train)
    m_pred_test <- m$predict(x = x_test)
    
    perf_train <- sapply(metricCls_all[c("IAM", "MCC", "P4", "ROCAUC")], \(f) {
      f(truth = y_train, pred = m_pred_train$pred, prob = m_pred_train$prob)
    })
    perf_test <- sapply(metricCls_all[c("IAM", "MCC", "P4", "ROCAUC")], \(f) {
      f(truth = y_test, pred = m_pred_test$pred, prob = m_pred_test$prob)
    })
  results[[method_name]][[epoch]] <- data.frame(
    method = method_name,
    epoch = epoch,
    split_type = rep(c("train", "test"), each = 4),
    score = c(perf_train, perf_test)
  )
  print(results[[method_name]][[epoch]])
  }
}

results_unlisted <- data.frame(do.call(rbind, lapply(results, \(m) do.call(rbind, m))), row.names = NULL)
results_unlisted$metric <- c("IAM", "MCC", "P4", "ROCAUC")
colnames(results_unlisted)[4] <- "score"

plt_learningCurve <- ggplot(results_unlisted) +
  geom_smooth(mapping = aes(
    x = epoch,
    y = score,
    color = method
  ), linewidth = 0.5) +
  facet_grid(split_type ~ metric) +
  ggsci::scale_color_d3(name = "", label = c(
    "BalNN", "CoverNet", "FocalNN", "FocalSDNN", "LDAMNN", "MMNN", "NN", "SMOTE + NN"
  )) +
  scale_x_continuous(name = "Epoch") +
  scale_y_continuous(name = "Score") +
  theme_bw()


ggsave(filename = "results/Figures/learningCurve.png",
       width = 12, height = 5, dpi = 300)

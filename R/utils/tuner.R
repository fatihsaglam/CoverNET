#' Tune Resampler Hyperparameters
#' @param res_name Name of the resampler from factory
#' @param data Full data.frame (last col is target)
#' @param n_reps Number of CV repeats (default 1)
#' @param k Number of folds (default 3)
#' @param n_iter Random search budget
#' @return A list (best_params)
tune_resampler <- function(res_name, data, n_reps = 1, k = 3, n_iter = 15) {
  res_info <- get_resampler(res_name)
  grid <- res_info$hypGrid
  
  # Random Search Selection
  search_indices <- sample(1:nrow(grid), min(nrow(grid), n_iter))
  candidate_configs <- grid[search_indices, , drop = FALSE]
  
  target_col <- ncol(data)
  y_vec <- data[[target_col]]
  
  # Storage for config performance
  config_results <- numeric(nrow(candidate_configs))
  
  for (c_idx in 1:nrow(candidate_configs)) {
    current_params <- as.list(candidate_configs[c_idx, , drop = FALSE])
    rep_scores <- numeric(n_reps)
    
    for (r in 1:n_reps) {
      folds <- caret::createFolds(y_vec, k = k)
      fold_scores <- numeric(k)
      
      for (f in 1:k) {
        # Split
        train_idx <- setdiff(1:nrow(data), folds[[f]])
        train_set <- data[train_idx, ]
        val_set   <- data[folds[[f]], ]
        
        # 1. Preprocess
        prep <- DataPreprocessor$new()
        # fit_transform returns x_train_clean
        x_train_clean <- prep$fit_transform(train_set[, -target_col], train_set[[target_col]])
        x_val_clean   <- prep$transform(val_set[, -target_col])

        # 2. Resample
        res_args <- c(list(x = x_train_clean, y = train_set[[target_col]]), current_params)
        saveRDS(res_args, "last_res_args.rds")
        res_data <- do.call(res_info$fun, res_args)
        
        # 3. Proxy Model (Decision Tree)
        # We need probabilities for ROCAUC
        fit <- rpart::rpart(y ~ ., data = data.frame(res_data$x_new, y = res_data$y_new), method = "class")
        probs <- predict(fit, x_val_clean, type = "prob")
        
        # 4. Score using metricCls_ROCAUC
        fold_scores[f] <- metricCls_ROCAUC(truth = val_set[[target_col]], prob = probs)
      }
      rep_scores[r] <- mean(fold_scores)
    }
    config_results[c_idx] <- mean(rep_scores)
  }
  
  torch$cuda$empty_cache()
  
  # Return the best row as a list
  best_idx <- which.max(config_results)
  return(as.list(candidate_configs[best_idx, , drop = FALSE]))
}
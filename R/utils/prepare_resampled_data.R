#' Prepare Resampled Folds for Evaluation (Flat Version)
#' @return A flat list of folds containing train_res, test_clean, and runtime
prepare_resampled_data <- function(res_name, data, best_params, n_reps = 1, k = 10) {
  res_info <- get_resampler(res_name)
  res_fun <- res_info$fun
  target_col <- ncol(data)
  y_vec <- data[[target_col]]
  
  # createMultiFolds generates a flat list of indices for the TRAINING sets
  # format: "Fold1.Rep1", "Fold2.Rep1", etc.
  multi_folds <- caret::createMultiFolds(y_vec, k = k, times = n_reps)
  
  # Pre-allocate results list for speed
  out_list <- vector("list", length(multi_folds))
  
  for (i in seq_along(multi_folds)) {
    # createMultiFolds returns TRAINING indices by default
    train_idx <- multi_folds[[i]]
    test_idx  <- setdiff(seq_len(nrow(data)), train_idx)
    
    train_set <- data[train_idx, ]
    test_set  <- data[test_idx, ]
    
    # 1. Preprocess: Fit on Train, Transform both
    # This prevents "Data Leakage" from the test set into training
    prep <- DataPreprocessor$new()
    x_train_clean <- prep$fit_transform(train_set[, -target_col], train_set[[target_col]])
    x_test_clean  <- prep$transform(test_set[, -target_col])
    
    # 2. Resample with Runtime tracking
    start_t <- Sys.time()
    res_args <- c(list(x = x_train_clean, y = train_set[[target_col]]), best_params)
    res_data <- do.call(res_fun, res_args)
    end_t <- Sys.time()
    
    runtime <- as.numeric(difftime(end_t, start_t, units = "secs"))
    
    # Extract Fold and Repeat info from the list names (e.g., "Fold01.Rep1")
    fold_name <- names(multi_folds)[i]
    colnames(res_data$x_new) <- colnames(x_test_clean)
    out_list[[i]] <- list(
      train_res = data.frame(res_data$x_new, y = res_data$y_new),
      test_clean = data.frame(x_test_clean, y = test_set[[target_col]]),
      runtime = runtime,
      id = fold_name
    )
  }
  
  return(out_list)
}
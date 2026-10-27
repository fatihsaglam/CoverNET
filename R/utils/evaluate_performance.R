#' Evaluate Classifiers on Prepared Resampled Data (Flat Version)
#' 
#' @param prepared_data A flat list of folds (output from prepare_resampled_data)
#' @param classifiers A named list of initialized R6 classifier objects
#' @param metrics A character vector of names (e.g., c("ROCAUC", "F1"))
#' @return A data frame with performance results per fold/classifier
evaluate_performance <- function(prepared_data, classifiers, metrics = c("ROCAUC", "F1", "GMEAN")) {
  results_list <- list()
  
  # Loop through the flat list of folds
  for (i in seq_along(prepared_data)) {
    
    fold_data <- prepared_data[[i]]
    fold_id   <- fold_data$id # "Fold01.Rep1" etc.
    
    # Separate features (X) and target (y) for the test set
    # We use the 'y' column explicitly to avoid index errors
    test_X <- fold_data$test_clean[, -ncol(fold_data$test_clean), drop = FALSE]
    test_y <- fold_data$test_clean$y
    
    # Separate features (X) and target (y) for the resampled training set
    train_X <- fold_data$train_res[, -ncol(fold_data$train_res), drop = FALSE]
    train_y <- fold_data$train_res$y
    
    # Loop through each selected Classifier
    for (clf_name in names(classifiers)) {
      
      # Access the R6 instance
      clf_obj <- classifiers[[clf_name]]
      
      # 1. TRAIN
      clf_obj$fit(x = train_X, y = train_y)
      
      # 2. PREDICT
      # Returns list(pred = factor, prob = matrix/vector)
      preds <- clf_obj$predict(test_X)
      
      # 3. CALCULATE METRICS
      # sapply returns a named vector of scores
      fold_metrics <- sapply(metrics, function(m_name) {
        metric_fn <- metricCls_all[[m_name]]
        
        perf <- metric_fn(
          truth = test_y, 
          pred  = preds$pred, 
          prob  = preds$prob
        )
        perf <- unname(perf)
        return(perf)
      })
      
      # 4. STORE RESULTS
      # We include the runtime from the resampling step and the unique Fold ID
      res_row <- data.frame(
        Fold_ID    = fold_id,
        Classifier = clf_name,
        Resample_Time = fold_data$runtime,
        as.list(fold_metrics),
        stringsAsFactors = FALSE
      )
      
      results_list[[length(results_list) + 1]] <- res_row
    }
    
    # Optional: Print progress to console for long runs
    # cat(sprintf("Processed Fold %d/%d\n", i, length(prepared_data)))
  }
  
  torch$cuda$empty_cache()
  
  # Combine all rows into one master results table
  final_results <- do.call(rbind, results_list)
  return(final_results)
}
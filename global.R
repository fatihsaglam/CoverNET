# ==============================================================================
# Script: global.R
# Purpose: Load libraries, source files, load data, set global settings
# ==============================================================================

# ==============================================================================
# 1. LIBRARIES
# ==============================================================================
library(R6)
library(reticulate)
library(ggplot2)
library(SMOTEWB)
library(randomForest)
library(caret)
library(yardstick)
library(kernlab)
library(crayon)
library(openxlsx)
library(mlr)
library(e1071)
library(MASS)

# ==============================================================================
# 2. SOURCE ALL R FILES
# ==============================================================================
r_files <- list.files("R", pattern = "\\.[Rr]$", recursive = TRUE, full.names = TRUE)
invisible(lapply(r_files, source))

py_run_string("
import torch
import gc

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
")

# ==============================================================================
# 3. GLOBAL SETTINGS
# ==============================================================================
SEED         <- 42
N_FOLDS      <- 5
TUNE_METRIC  <- "ROCAUC"
EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4")

set.seed(SEED)

# ==============================================================================
# 4. LOAD DATA
# ==============================================================================
datasets_binary <- readRDS(file = "data/datasets_binary.rds")
dataset_names   <- names(datasets_binary)
datasets_info   <- attr(datasets_binary, "info")

cat(green(sprintf("Loaded %d datasets\n", length(datasets_binary))))

# ==============================================================================
# 5. FLAT CV FOLD CREATION
# ==============================================================================

make_folds <- function(y, k = N_FOLDS, seed = SEED) {
  set.seed(seed)
  n       <- length(y)
  classes <- levels(y)
  folds   <- vector("list", k)
  
  for (i in 1:k) folds[[i]] <- list(train = integer(0), test = integer(0))
  
  for (cls in classes) {
    idx   <- which(y == cls)
    idx   <- sample(idx)
    cuts  <- cut(seq_along(idx), breaks = k, labels = FALSE)
    for (i in 1:k) {
      test_idx  <- idx[cuts == i]
      train_idx <- idx[cuts != i]
      folds[[i]]$test  <- c(folds[[i]]$test,  test_idx)
      folds[[i]]$train <- c(folds[[i]]$train, train_idx)
    }
  }
  folds
}

# ==============================================================================
# 6. SINGLE FOLD EVALUATION
# ==============================================================================

eval_fold <- function(method, x, y, fold, params = list(),
                      fold_id = NULL, n_folds = N_FOLDS,
                      config_id = NULL, n_configs = NULL,
                      verbose = TRUE) {
  
  tryCatch({
    
    # --- Progress message ---
    if (verbose) {
      fold_str   <- if (!is.null(fold_id))   sprintf("Fold %d/%d", fold_id, n_folds)   else "Fold"
      config_str <- if (!is.null(config_id) && !is.null(n_configs))
        sprintf(" | Config %d/%d", config_id, n_configs) else ""
      cat(silver(sprintf("      [%s%s]\n", fold_str, config_str)))
    }
    
    # --- Split ---
    x_train <- x[fold$train, , drop = FALSE]
    y_train <- y[fold$train]
    x_test  <- x[fold$test,  , drop = FALSE]
    y_test  <- y[fold$test]
    
    # --- Preprocess ---
    prep    <- DataPreprocessor$new()
    x_train <- prep$fit_transform(x_train, y_train, scale_method = "z-score")
    x_test  <- prep$transform(x_test)
    
    # --- Resample ---
    resampler_fn <- resamplers[[method$resampler]]$fn
    resampled    <- resampler_fn(x_train, y_train)
    x_train      <- resampled$x
    y_train      <- resampled$y
    
    # --- Combine params ---
    all_params <- c(method$fixed_params, params)
    
    # --- Fit + Predict with timing ---
    clf     <- method$classifier_class$new()
    t_start <- proc.time()["elapsed"]
    do.call(clf$fit, c(list(x = x_train, y = y_train), all_params))
    result  <- clf$predict(x_test)
    t_end   <- proc.time()["elapsed"]
    runtime <- as.numeric(t_end - t_start)
    
    pred <- result$pred
    prob <- result$prob
    
    # --- Evaluate metrics ---
    scores <- sapply(EVAL_METRICS, function(m) {
      tryCatch(
        metricCls_all[[m]](truth = y_test, pred = pred, prob = prob),
        error = function(e) NA_real_
      )
    })
    
    # --- Memory flush ---
    py_run_string("flush_memory()")
    
    c(scores, Runtime = runtime)
    
  }, error = function(e) {
    message(red(sprintf("      Fold error: %s", e$message)))
    rep(NA_real_, length(EVAL_METRICS) + 1L)
  })
}

# ==============================================================================
# 7. FLAT CV TUNING + EVALUATION
# ==============================================================================

run_flat_cv <- function(method, x, y, folds, verbose = TRUE) {
  
  all_metric_names <- c(EVAL_METRICS, "Runtime")
  n_folds          <- length(folds)
  
  # ---- Case 1: No tuning ----
  if (!method$tuning) {
    
    if (verbose) cat(silver("    No tuning — running CV folds\n"))
    
    scores_mat <- do.call(rbind, lapply(seq_along(folds), function(i) {
      eval_fold(method, x, y, folds[[i]],
                params    = list(),
                fold_id   = i,
                n_folds   = n_folds,
                verbose   = verbose)
    }))
    
    return(list(
      best_params = method$fixed_params,
      cv_scores   = scores_mat,
      mean_scores = colMeans(scores_mat, na.rm = TRUE),
      sd_scores   = apply(scores_mat, 2, sd, na.rm = TRUE)
    ))
  }
  
  # ---- Case 2: Tuning ----
  grid      <- method$tune_grid
  n_configs <- nrow(grid)
  
  if (verbose) cat(silver(sprintf(
    "    Tuning — %d configs x %d folds = %d fits\n",
    n_configs, n_folds, n_configs * n_folds
  )))
  
  all_config_results <- lapply(1:n_configs, function(g) {
    
    if (verbose) cat(yellow(sprintf(
      "    [Config %d/%d]\n", g, n_configs
    )))
    
    params <- as.list(grid[g, , drop = FALSE])
    
    do.call(rbind, lapply(seq_along(folds), function(i) {
      eval_fold(method, x, y, folds[[i]],
                params    = params,
                fold_id   = i,
                n_folds   = n_folds,
                config_id = g,
                n_configs = n_configs,
                verbose   = verbose)
    }))
  })
  
  # Select best config by mean TUNE_METRIC
  tune_means <- sapply(all_config_results, function(mat) {
    mean(mat[, TUNE_METRIC], na.rm = TRUE)
  })
  
  best_idx <- which.max(tune_means)
  best_mat <- all_config_results[[best_idx]]
  
  if (verbose) cat(green(sprintf(
    "    Best config: %d | %s: %.4f\n",
    best_idx,
    TUNE_METRIC,
    tune_means[best_idx]
  )))
  
  list(
    best_params = c(method$fixed_params,
                    as.list(grid[best_idx, , drop = FALSE])),
    cv_scores   = best_mat,
    mean_scores = colMeans(best_mat, na.rm = TRUE),
    sd_scores   = apply(best_mat, 2, sd, na.rm = TRUE)
  )
}

# ==============================================================================
# 8. HELPER: results list to data.frame
# ==============================================================================

results_to_df <- function(results) {
  
  all_metric_names <- c(EVAL_METRICS, "Runtime")
  ds_names         <- names(results)
  
  # Get full method list from first dataset (assumes consistent across datasets)
  method_names <- unique(unlist(lapply(results, names)))
  
  # --- Build mean and SD wide tables ---
  mean_rows <- list()
  sd_rows   <- list()
  
  for (ds_name in ds_names) {
    for (metric in all_metric_names) {
      
      mean_row <- list(Dataset = ds_name, Metric = metric)
      sd_row   <- list(Dataset = ds_name, Metric = metric)
      
      for (method_name in method_names) {
        res <- results[[ds_name]][[method_name]]
        if (is.null(res)) {
          mean_row[[method_name]] <- NA
          sd_row[[method_name]]   <- NA
        } else {
          mean_row[[method_name]] <- round(res$mean_scores[metric], 4)
          sd_row[[method_name]]   <- round(res$sd_scores[metric],   4)
        }
      }
      
      mean_rows[[length(mean_rows) + 1]] <- as.data.frame(mean_row,
                                                          stringsAsFactors = FALSE)
      sd_rows[[length(sd_rows) + 1]]     <- as.data.frame(sd_row,
                                                          stringsAsFactors = FALSE)
    }
  }
  
  # --- Build best params table ---
  param_rows <- list()
  for (ds_name in ds_names) {
    for (method_name in method_names) {
      res <- results[[ds_name]][[method_name]]
      if (is.null(res)) {
        param_str <- NA
      } else if (length(res$best_params) == 0) {
        param_str <- ""
      } else {
        param_str <- paste(
          names(res$best_params),
          unlist(res$best_params),
          sep = "=", collapse = ", "
        )
      }
      param_rows[[length(param_rows) + 1]] <- data.frame(
        Dataset    = ds_name,
        Method     = method_name,
        BestParams = param_str,
        stringsAsFactors = FALSE
      )
    }
  }
  
  list(
    means      = do.call(rbind, mean_rows),
    sds        = do.call(rbind, sd_rows),
    bestparams = do.call(rbind, param_rows)
  )
}

# ==============================================================================
# 9. HELPER: save results
# ==============================================================================

save_results <- function(results) {
  if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)
  
  out <- results_to_df(results)
  if (is.null(out$means) || nrow(out$means) == 0) return(invisible(NULL))
  
  saveRDS(results, file = file.path(RESULTS_DIR, "results.rds"))
  
  wb <- createWorkbook()
  
  addWorksheet(wb, "Means")
  writeData(wb, "Means", out$means)
  freezePane(wb, "Means", firstRow = TRUE, firstCol = TRUE)
  setColWidths(wb, "Means", cols = 1:ncol(out$means), widths = "auto")
  
  addWorksheet(wb, "SDs")
  writeData(wb, "SDs", out$sds)
  freezePane(wb, "SDs", firstRow = TRUE, firstCol = TRUE)
  setColWidths(wb, "SDs", cols = 1:ncol(out$sds), widths = "auto")
  
  addWorksheet(wb, "BestParams")
  writeData(wb, "BestParams", out$bestparams)
  freezePane(wb, "BestParams", firstRow = TRUE)
  setColWidths(wb, "BestParams", cols = 1:ncol(out$bestparams), widths = "auto")
  
  saveWorkbook(wb, file = file.path(RESULTS_DIR, "results.xlsx"),
               overwrite = TRUE)
  
  invisible(out)
}
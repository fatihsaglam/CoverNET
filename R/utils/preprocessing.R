# ==============================================================================
# Script: preprocessing.R
# Purpose: Stateful, leakage-free data preprocessing using R6
# ==============================================================================

library(R6)

#' DataPreprocessor R6 Class
#'
#' @description
#' A stateful preprocessor that learns scaling parameters and feature masks
#' strictly from training data to prevent data leakage during CV/testing.
#'
#' @export
DataPreprocessor <- R6::R6Class(
  "DataPreprocessor",
  
  public = list(
    
    #' @field params List holding the fitted parameters (means, sds, masks).
    params = NULL,
    
    #' @description
    #' Initialize a new preprocessor object.
    initialize = function() {
      self$params <- NULL
    },
    
    #' @description
    #' Fit the preprocessor to the training data.
    #' @param x_train A matrix or data.frame of training features.
    #' @param y_train A factor vector of training labels.
    #' @param min_class_samples Minimum required samples per class (default: 8).
    #' @param var_threshold Minimum variance allowed PER CLASS for a feature (default: 1e-4).
    #' @param scale_method String: "z-score", "min-max", or "none".
    #' @return The object itself (invisibly) for method chaining.
    fit = function(x_train, y_train, 
                   min_class_samples = 8, 
                   var_threshold = 1e-4, 
                   scale_method = c("z-score", "min-max", "none")) {
      
      scale_method <- match.arg(scale_method)
      
      if (!is.factor(y_train)) {
        y_train <- as.factor(y_train)
      }
      
      # 1. Class Size Safety Check
      class_counts <- table(y_train)
      if (any(class_counts < min_class_samples)) {
        min_cls <- names(class_counts)[which.min(class_counts)]
        stop(sprintf("Training class '%s' has only %d samples (minimum: %d).", 
                     min_cls, min(class_counts), min_class_samples))
      }
      
      # 2. Learn Features to Keep (Per-Class Variance Filter)
      keep_features <- rep(TRUE, ncol(x_train))
      
      for (cls in levels(y_train)) {
        idx <- which(y_train == cls)
        vars <- apply(x_train[idx, , drop = FALSE], 2, stats::var)
        
        # Feature must have variance > threshold in ALL classes
        keep_features <- keep_features & !is.na(vars) & (vars > var_threshold)
      }
      
      if (!any(keep_features)) {
        stop("All features removed after per-class variance filter.")
      }
      
      # Subset to only the valid features to calculate scaling parameters
      x_filtered <- x_train[, keep_features, drop = FALSE]
      
      # 3. Learn Scaling Parameters
      self$params <- list(
        keep_features = keep_features,
        scale_method = scale_method,
        center = NULL,
        scale = NULL
      )
      
      if (scale_method == "z-score") {
        
        self$params$center <- apply(x_filtered, 2, mean)
        sds <- apply(x_filtered, 2, stats::sd)
        sds[sds == 0] <- 1 # Prevent division by zero
        self$params$scale <- sds
        
      } else if (scale_method == "min-max") {
        
        self$params$center <- apply(x_filtered, 2, min)
        maxs <- apply(x_filtered, 2, max)
        ranges <- maxs - self$params$center
        ranges[ranges == 0] <- 1 # Prevent division by zero
        self$params$scale <- ranges
        
      }
      
      # Return self invisibly to allow method chaining (e.g., prep$fit()$transform())
      invisible(self)
    },
    
    #' @description
    #' Apply the learned parameters to a dataset.
    #' @param x A matrix or data.frame of features to transform.
    #' @return A preprocessed data.frame.
    transform = function(x) {
      
      if (is.null(self$params)) {
        stop("Preprocessor has not been fitted. Call $fit() first.")
      }
      
      # 1. Drop features that failed the variance check during training
      x_proc <- x[, self$params$keep_features, drop = FALSE]
      
      # 2. Apply the exact scaling learned from training
      if (self$params$scale_method != "none") {
        x_proc <- sweep(x_proc, 2, self$params$center, FUN = "-")
        x_proc <- sweep(x_proc, 2, self$params$scale, FUN = "/")
      }
      
      return(as.data.frame(x_proc))
    },
    
    #' @description
    #' Convenience method to fit the model and transform the training data in one step.
    #' @param x_train A matrix or data.frame of training features.
    #' @param y_train A factor vector of training labels.
    #' @param ... Additional arguments passed to $fit()
    #' @return A preprocessed data.frame.
    fit_transform = function(x_train, y_train, ...) {
      self$fit(x_train, y_train, ...)
      return(self$transform(x_train))
    }
  )
)
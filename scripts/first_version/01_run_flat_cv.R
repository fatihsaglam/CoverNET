# ==============================================================================
# Script: run.R
# Purpose: Main experiment loop
# ==============================================================================

source("global.R")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# ==============================================================================
# 1. SETTINGS
# ==============================================================================

SEED         <- 42
N_FOLDS      <- 5
TUNE_METRIC  <- "ROCAUC"
EVAL_METRICS <- c("ROCAUC", "PRAUC", "GMEAN", "F1", "MCC", "BACC", "P4", "IAM", "BBrier")

# --- Dataset selection ---
# Use NULL to run all, or provide integer indexes or names
DATASET_IDX  <- NULL         # e.g. c(1, 5, 10) or NULL for all
DATASET_IDX  <- 133:length(datasets_binary)         # e.g. c(1, 5, 10) or NULL for all
# DATASET_IDX <- which(datasets_info$n*datasets_info$p < 25000)

DATASET_NAMES <- NULL        # e.g. c("ecoli", "pima") or NULL for all

# --- Method selection ---
# Use NULL to run all, or provide integer indexes or names
METHOD_IDX   <- NULL         # e.g. c(1, 3, 5) or NULL for all
METHOD_NAMES <- NULL         # e.g. c("RF", "NNET", "CCCD_NNET") or NULL for all

# --- Output ---
RESULTS_DIR  <- "results"
SAVE_PARTIAL <- TRUE         # Save after each dataset
VERBOSE      <- TRUE         # Print progress

# ==============================================================================
# 2. RESOLVE SELECTIONS
# ==============================================================================

all_dataset_names <- names(datasets_binary)
selected_datasets <- if (!is.null(DATASET_NAMES)) DATASET_NAMES else
  if (!is.null(DATASET_IDX))   all_dataset_names[DATASET_IDX] else
    all_dataset_names

all_method_names  <- names(methods_list)
selected_methods  <- if (!is.null(METHOD_NAMES)) METHOD_NAMES else
  if (!is.null(METHOD_IDX))   all_method_names[METHOD_IDX] else
    all_method_names

cat(green(sprintf(
  "Running %d datasets x %d methods\n",
  length(selected_datasets),
  length(selected_methods)
)))



# ==============================================================================
# 5. MAIN LOOP
# ==============================================================================

results <- list()

for (ds_name in selected_datasets) {
  
  if (VERBOSE) cat(cyan(sprintf("\n[Dataset: %s]\n", ds_name)))
  
  dataset <- datasets_binary[[ds_name]]
  x       <- as.matrix(dataset[, -ncol(dataset)])
  y       <- as.factor(dataset[,  ncol(dataset)])
  folds   <- make_folds(y, k = N_FOLDS, seed = SEED)
  
  if (any(table(y) < 15)) {
    cat("low number of samples in positive class (", min(table(y)), ")\n")
    next
  }
  
  results[[ds_name]] <- list()
  
  for (method_name in selected_methods) {
    
    if (VERBOSE) cat(sprintf("  [Method: %-20s] ... ", method_name))
    
    t_start <- proc.time()["elapsed"]
    
    result <- tryCatch(
      run_flat_cv(method = methods_list[[method_name]],
                  x = x, y = y, folds = folds),
      error = function(e) {
        message(red(sprintf("FAILED: %s", e$message)))
        NULL
      }
    )
    
    t_end <- proc.time()["elapsed"]
    
    results[[ds_name]][[method_name]] <- result
    
    if (VERBOSE) {
      if (!is.null(result)) {
        cat(green(sprintf("OK [%.1fs] | %s: %.4f\n",
                          as.numeric(t_end - t_start),
                          TUNE_METRIC,
                          result$mean_scores[TUNE_METRIC])))
      } else {
        cat(red("FAILED\n"))
      }
    }
  }
  
  if (SAVE_PARTIAL) {
    save_results(results)
    if (VERBOSE) cat(yellow(sprintf(
      "  Saved (%d/%d)\n",
      which(selected_datasets == ds_name),
      length(selected_datasets)
    )))
  }
}

# ==============================================================================
# 6. FINAL SAVE
# ==============================================================================

save_results(results)
cat(green("\nDone.\n"))
# ==============================================================================
# Script: factory.R
# Purpose: Define all methods and their hyperparameter grids for the experiment
# ==============================================================================

# ==============================================================================
# 1. RESAMPLER FACTORY
# ==============================================================================

resamplers <- list(
  
  none = list(
    name = "none",
    fn   = function(x, y) list(x = x, y = y)
  ),
  
  SMOTE = list(
    name = "SMOTE",
    fn   = function(x, y) {
      result <- SMOTEWB::SMOTE(x = as.data.frame(x), y = y)
      list(x = result$x_new, y = result$y_new)
    }
  ),
  
  RUS = list(
    name = "RUS",
    fn   = function(x, y) {
      result <- SMOTEWB::RUS(x = as.data.frame(x), y = y)
      list(x = result$x_new, y = result$y_new)
    }
  )
)

# ==============================================================================
# 2. HELPER: Build a method entry
# ==============================================================================

make_method <- function(name,
                        classifier_class,
                        resampler    = "none",
                        tuning       = FALSE,
                        fixed_params = list(),
                        tune_grid    = NULL) {
  list(
    name             = name,
    classifier_class = classifier_class,
    resampler        = resampler,
    tuning           = tuning,
    fixed_params     = fixed_params,
    tune_grid        = tune_grid
  )
}

# ==============================================================================
# 3. RANDOM SEARCH GRIDS
# ==============================================================================

N_RANDOM <- 20L
set.seed(42)

# Fixed NNET training settings shared across all NNETs
nnet_fixed <- list(
  activation = "relu",
  epochs     = 100L,
  batch_size = 256L,
  lr         = 1e-3,
  patience   = 10L
)

# --- Shared architecture sampler ---
sample_arch <- function(n) {
  data.frame(
    hidden_dim = sample(c(16L, 32L, 64L, 128L), n, replace = TRUE),
    n_hidden   = sample(c(1L, 2L, 3L), n, replace = TRUE)
  )
}

# --- NNET (architecture only) ---
nnet_arch_grid <- sample_arch(N_RANDOM)

# --- Focal ---
focal_grid <- cbind(
  sample_arch(N_RANDOM),
  data.frame(
    gamma = runif(N_RANDOM, min = 0.5, max = 5.0)
  )
)

# --- FocalSD ---
focalsd_grid <- cbind(
  sample_arch(N_RANDOM),
  data.frame(
    gamma_low  = runif(N_RANDOM, min = 2.0, max = 7.0),
    gamma_high = runif(N_RANDOM, min = 0.5, max = 4.0)
  )
)

# --- LDAM ---
ldam_grid <- cbind(
  sample_arch(N_RANDOM),
  data.frame(
    C        = runif(N_RANDOM, min = 0.1, max = 2.0),
    reweight = sample(c(TRUE, FALSE), N_RANDOM, replace = TRUE)
  )
)

# --- MM ---
mm_grid <- cbind(
  sample_arch(N_RANDOM),
  data.frame(
    delta_pos = runif(N_RANDOM, min = 0.1, max = 1.5),
    delta_neg = runif(N_RANDOM, min = 0.1, max = 1.5)
  )
)

# --- CCCD ---
cccd_grid <- cbind(
  sample_arch(N_RANDOM),
  data.frame(
    lambda1 = runif(N_RANDOM, min = 0.1, max = 2.0),
    lambda2 = runif(N_RANDOM, min = 0.1, max = 2.0),
    lambda3 = runif(N_RANDOM, min = 0.001, max = 0.5)
  )
)

# --- PNN ---
pnn_grid <- data.frame(
  sigma = runif(N_RANDOM, min = 0.05, max = 3.0)
)

# --- SkewPNN ---
skewpnn_grid <- data.frame(
  sigma = runif(N_RANDOM, min = 0.05, max = 3.0),
  alpha = runif(N_RANDOM, min = -7.0, max = -0.5)
)

# --- CCCD K grid (revision 1): K_c = min(n_c, max(K_MIN, floor(alpha * sqrt(n_c)))); alpha = 1 is the sqrt rule ---
# Used by 'revision 1 - run.R' (alpha tuned on the best configuration of every dataset); cccd_grid itself is unchanged.
ALPHA_LEVELS <- c(0.25, 0.5, 1)
K_MIN        <- 2L

# --- RF and SVM families, tuned (revision 1, reviewer 3 comment 5): the same budget of N_RANDOM configurations ---
# Separate seed, so that every grid above stays unchanged. mtry = round(mtry_frac * p); sigma = sigma_scale / p.
set.seed(2026)
rf_grid  <- data.frame(
  mtry_frac     = runif(N_RANDOM, min = 0.05, max = 1.0),
  min.node.size = sample(c(1L, 3L, 5L, 10L, 20L), N_RANDOM, replace = TRUE)
)
svm_grid <- data.frame(
  C           = 2^runif(N_RANDOM, min = -3, max = 7),
  sigma_scale = 2^runif(N_RANDOM, min = -4, max = 2)
)

# ==============================================================================
# 4. METHOD LIST
# ==============================================================================

methods_list <- list(
  
  # --- Classical Classifiers (no tuning) ---
  RF = make_method(
    name             = "RF",
    classifier_class = classifier_RF,
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  NaiveBayes = make_method(
    name             = "NaiveBayes",
    classifier_class = classifier_NaiveBayes,
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  CART = make_method(
    name             = "CART",
    classifier_class = classifier_CART,
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  SVMRadial = make_method(
    name             = "SVMRadial",
    classifier_class = classifier_SVMRadial,
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  KNN = make_method(
    name             = "KNN",
    classifier_class = classifier_KNN,
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  LogReg = make_method(
    name             = "LogReg",
    classifier_class = classifier_LogReg,
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  # --- Resampling Methods (no tuning) ---
  SMOTE_RF = make_method(
    name             = "SMOTE_RF",
    classifier_class = classifier_RF,
    resampler        = "SMOTE",
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  RUS_RF = make_method(
    name             = "RUS_RF",
    classifier_class = classifier_RF,
    resampler        = "RUS",
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  SMOTE_SVM = make_method(
    name             = "SMOTE_SVM",
    classifier_class = classifier_SVMRadial,
    resampler        = "SMOTE",
    tuning           = FALSE,
    fixed_params     = list()
  ),
  
  # --- Standard NNETs (tuning: architecture only) ---
  NNET = make_method(
    name             = "NNET",
    classifier_class = classifier_NNET,
    tuning           = TRUE,
    fixed_params     = c(nnet_fixed, list(balanced = FALSE)),
    tune_grid        = nnet_arch_grid
  ),
  
  BalancedNNET = make_method(
    name             = "BalancedNNET",
    classifier_class = classifier_NNET,
    tuning           = TRUE,
    fixed_params     = c(nnet_fixed, list(balanced = TRUE)),
    tune_grid        = nnet_arch_grid
  ),
  
  SMOTENNET = make_method(
    name             = "SMOTENNET",
    classifier_class = classifier_NNET,
    resampler        = "SMOTE",
    tuning           = TRUE,
    fixed_params     = c(nnet_fixed, list(balanced = FALSE)),
    tune_grid        = nnet_arch_grid
  ),
  
  # --- NN Imbalance Methods (tuning: architecture + loss hyperparameters) ---
  FocalNNET = make_method(
    name             = "FocalNNET",
    classifier_class = classifier_FocalNNET,
    tuning           = TRUE,
    fixed_params     = nnet_fixed,
    tune_grid        = focal_grid
  ),
  
  FocalSDNNET = make_method(
    name             = "FocalSDNNET",
    classifier_class = classifier_FocalSDNNET,
    tuning           = TRUE,
    fixed_params     = c(nnet_fixed, list(threshold = 0.2)),
    tune_grid        = focalsd_grid
  ),
  
  LDAMNNET = make_method(
    name             = "LDAMNNET",
    classifier_class = classifier_LDAMNNET,
    tuning           = TRUE,
    fixed_params     = nnet_fixed,
    tune_grid        = ldam_grid
  ),
  
  MMNNET = make_method(
    name             = "MMNNET",
    classifier_class = classifier_MMNNET,
    tuning           = TRUE,
    fixed_params     = nnet_fixed,
    tune_grid        = mm_grid
  ),
  
  # --- Kernel Methods (tuning: kernel hyperparameters) ---
  PNN = make_method(
    name             = "PNN",
    classifier_class = classifier_PNNET,
    tuning           = TRUE,
    fixed_params     = list(),
    tune_grid        = pnn_grid
  ),
  
  SkewPNN = make_method(
    name             = "SkewPNN",
    classifier_class = classifier_SkewPNNET,
    tuning           = TRUE,
    fixed_params     = list(),
    tune_grid        = skewpnn_grid
  ),
  
  # --- Novel Method (tuning: architecture + CCCD hyperparameters) ---
  CCCDNNET = make_method(
    name             = "CCCDNNET",
    classifier_class = classifier_CCCDNNET,
    tuning           = TRUE,
    fixed_params     = nnet_fixed,
    tune_grid        = cccd_grid
  )
)

# --- Tuned RF and SVM families (revision 1, reviewer 3 comment 5): same random-search budget as the neural methods ---
methods_list$RFtuned       <- make_method(name = "RFtuned",       classifier_class = classifier_RF,
                                          tuning = TRUE, tune_grid = rf_grid)
methods_list$SMOTERFtuned  <- make_method(name = "SMOTERFtuned",  classifier_class = classifier_RF, resampler = "SMOTE",
                                          tuning = TRUE, tune_grid = rf_grid)
methods_list$RUSRFtuned    <- make_method(name = "RUSRFtuned",    classifier_class = classifier_RF, resampler = "RUS",
                                          tuning = TRUE, tune_grid = rf_grid)
methods_list$SVMtuned      <- make_method(name = "SVMtuned",      classifier_class = classifier_SVMRadial,
                                          tuning = TRUE, tune_grid = svm_grid)
methods_list$SMOTESVMtuned <- make_method(name = "SMOTESVMtuned", classifier_class = classifier_SVMRadial, resampler = "SMOTE",
                                          tuning = TRUE, tune_grid = svm_grid)

# --- Matched-head comparison (revision 1, reviewer 1 comment 2 / reviewer 3 comment 3): same hidden layers, same BCE loss,
# same architecture grid as NNET (nnet_arch_grid), only the output layer differs. CCLBCENNET = CoverNET head trained with
# the classification term only (lambda1 = lambda2 = lambda3 = 0), architecture tuned like NNET.
methods_list$MLPheadNNET <- make_method(name = "MLPheadNNET", classifier_class = classifier_HeadNNET, tuning = TRUE,
                                        fixed_params = c(nnet_fixed, list(head = "mlp")), tune_grid = nnet_arch_grid)
methods_list$RBFheadNNET <- make_method(name = "RBFheadNNET", classifier_class = classifier_HeadNNET, tuning = TRUE,
                                        fixed_params = c(nnet_fixed, list(head = "rbf")), tune_grid = nnet_arch_grid)
methods_list$CCLBCENNET  <- make_method(name = "CCLBCENNET",  classifier_class = classifier_CCCDNNET, tuning = TRUE,
                                        fixed_params = c(nnet_fixed, list(lambda1 = 0, lambda2 = 0, lambda3 = 0)),
                                        tune_grid = nnet_arch_grid)

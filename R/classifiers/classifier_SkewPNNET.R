library(R6)
library(reticulate)

py_run_string("
import numpy as np
from scipy.stats import norm

# SkewPNN: stores training data, computes skew-normal kernel at prediction time
# No gradient-based training — single-pass like all PNNs

def skew_kernel(x, xi, sigma, alpha):
    # x  : (d,) input sample
    # xi : (N, d) training samples
    diff = xi - x                          # (N, d)
    dist = np.linalg.norm(diff, axis=1)    # (N,)
    gauss = np.exp(-(dist**2) / (2 * sigma**2))
    skew  = norm.cdf(alpha * dist / sigma)
    return gauss * skew                    # (N,)

def skewpnn_predict(x_train, y_train, x_test, sigma, alpha):
    classes = np.unique(y_train)
    n_test  = x_test.shape[0]
    probs   = np.zeros((n_test, len(classes)))

    for i, xq in enumerate(x_test):
        class_scores = []
        for c in classes:
            mask = (y_train == c)
            xi_c = x_train[mask]
            k    = skew_kernel(xq, xi_c, sigma, alpha)
            class_scores.append(np.mean(k))   # mean kernel per class
        total = sum(class_scores) + 1e-12
        probs[i] = [s / total for s in class_scores]

    return probs   # (n_test, n_classes)
")

classifier_SkewPNNET <- R6Class(
  "classifier_SkewPNNET",
  public = list(
    x_train     = NULL,
    y_train_bin = NULL,
    class_names = NULL,
    class_pos   = NULL,
    sigma       = NULL,
    alpha       = NULL,
    
    initialize = function() {
      invisible(self)
    },
    
    fit = function(x, y, sigma = 0.2, alpha = -2.0) {
      self$sigma       <- sigma
      self$alpha       <- alpha
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))  # minority = positive class
      
      self$x_train     <- as.matrix(x)
      self$y_train_bin <- as.integer(y == self$class_pos)  # 1 = minority, 0 = majority
      
      invisible(self)
    },
    
    predict = function(x) {
      probs_mat <- py$skewpnn_predict(
        x_train  = self$x_train,
        y_train  = self$y_train_bin,
        x_test   = as.matrix(x),
        sigma    = self$sigma,
        alpha    = self$alpha
      )
      # probs_mat columns: [0=majority, 1=minority]
      prob <- matrix(NA, nrow = nrow(probs_mat), ncol = 2)
      pos_col <- which(self$class_names == self$class_pos)
      neg_col <- which(self$class_names != self$class_pos)
      prob[, pos_col] <- probs_mat[, 2]  # minority prob
      prob[, neg_col] <- probs_mat[, 1]  # majority prob
      colnames(prob) <- self$class_names
      
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      list(pred = pred, prob = prob)
    }
  )
)
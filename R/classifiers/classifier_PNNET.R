library(R6)
library(reticulate)

py_run_string("
import numpy as np

def gaussian_kernel(x, xi, sigma):
    # x  : (d,) input sample
    # xi : (N, d) training samples
    diff = xi - x                          # (N, d)
    dist_sq = np.sum(diff**2, axis=1)      # (N,)
    return np.exp(-dist_sq / (2 * sigma**2))

def pnn_predict(x_train, y_train, x_test, sigma):
    classes = np.unique(y_train)
    n_test  = x_test.shape[0]
    probs   = np.zeros((n_test, len(classes)))

    for i, xq in enumerate(x_test):
        class_scores = []
        for c in classes:
            mask = (y_train == c)
            xi_c = x_train[mask]
            k    = gaussian_kernel(xq, xi_c, sigma)
            class_scores.append(np.mean(k))
        total = sum(class_scores) + 1e-12
        probs[i] = [s / total for s in class_scores]

    return probs  # (n_test, n_classes)
")

classifier_PNNET <- R6Class(
  "classifier_PNNET",
  public = list(
    x_train     = NULL,
    y_train_bin = NULL,
    class_names = NULL,
    class_pos   = NULL,
    sigma       = NULL,
    
    initialize = function() {
      invisible(self)
    },
    
    fit = function(x, y, sigma = 0.2) {
      self$sigma       <- sigma
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      
      self$x_train     <- as.matrix(x)
      self$y_train_bin <- as.integer(y == self$class_pos)
      
      invisible(self)
    },
    
    predict = function(x) {
      probs_mat <- py$pnn_predict(
        x_train = self$x_train,
        y_train = self$y_train_bin,
        x_test  = as.matrix(x),
        sigma   = self$sigma
      )
      prob <- matrix(NA, nrow = nrow(probs_mat), ncol = 2)
      pos_col <- which(self$class_names == self$class_pos)
      neg_col <- which(self$class_names != self$class_pos)
      prob[, pos_col] <- probs_mat[, 2]
      prob[, neg_col] <- probs_mat[, 1]
      colnames(prob) <- self$class_names
      
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      list(pred = pred, prob = prob)
    }
  )
)
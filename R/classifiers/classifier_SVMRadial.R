library(R6)
library(kernlab)
library(e1071)

classifier_SVMRadial <- R6Class("classifier_SVMRadial",
                                public = list(
                                  model = NULL,
                                  method = NULL,
                                  var_names = NULL,
                                  class_names = NULL,
                                  
                                  fit = function(x, y, C = 1, sigma_scale = NULL, ...) {
                                    x_mat <- as.matrix(x)
                                    self$var_names <- colnames(x_mat)
                                    self$class_names <- levels(y)
                                    
                                    kpar  <- if (is.null(sigma_scale)) "automatic" else list(sigma = sigma_scale / ncol(x_mat))
                                    gamma <- (if (is.null(sigma_scale)) 1 else sigma_scale) / ncol(x_mat)
                                    tol <- 1e-3
                                    self$method <- "kernlab"
                                    
                                    repeat {
                                      # Try kernlab with RBF kernel
                                      m <- try(kernlab::ksvm(
                                        x = x_mat,
                                        y = y,
                                        kernel = "rbfdot",
                                        kpar = kpar,
                                        C = C,
                                        prob.model = TRUE,
                                        tol = tol,
                                        ...
                                      ), silent = TRUE)
                                      
                                      if (inherits(m, "try-error")) {
                                        tol <- tol * 10
                                        if (tol > 1) {
                                          # Fallback to e1071 radial
                                          self$method <- "e1071"
                                          self$model <- e1071::svm(
                                            x = x_mat, y = y, 
                                            kernel = "radial", 
                                            cost = C, gamma = gamma,
                                            probability = TRUE, 
                                            ...
                                          )
                                          break
                                        }
                                      } else {
                                        if (all(sapply(m@prob.model, function(mm) length(mm) == 2))) {
                                          self$model <- m
                                          break
                                        } else {
                                          tol <- tol * 10
                                        }
                                      }
                                    }
                                    invisible(self)
                                  },
                                  
                                  predict = function(x) {
                                    x_mat <- as.matrix(x)
                                    if (self$method == "kernlab") {
                                      prob <- predict(self$model, newdata = x_mat, type = "probabilities")
                                    } else {
                                      pred_obj <- predict(self$model, newdata = x_mat, probability = TRUE)
                                      prob <- attr(pred_obj, "probabilities")
                                    }
                                    
                                    # Standardize order and output
                                    prob <- prob[, self$class_names, drop = FALSE]
                                    pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
                                    
                                    return(list(pred = pred, prob = prob))
                                  }
                                )
)
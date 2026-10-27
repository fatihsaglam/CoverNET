library(R6)
library(kernlab)
library(e1071)

classifier_SVMLinear <- R6Class("SVMLinear",
                           public = list(
                             model = NULL,
                             method = NULL, # Stores which package was eventually used
                             var_names = NULL,
                             class_names = NULL,
                             
                             fit = function(x, y, ...) {
                               x_mat <- as.matrix(x)
                               self$var_names <- colnames(x_mat)
                               self$class_names <- levels(y)
                               
                               tol <- 1e-3
                               self$method <- "kernlab"
                               
                               repeat {
                                 # Attempt kernlab (ksvm)
                                 m <- try(kernlab::ksvm(
                                   x = x_mat,
                                   y = y,
                                   kernel = "vanilladot",
                                   kpar = list(),
                                   prob.model = TRUE,
                                   tol = tol,
                                   ...
                                 ), silent = TRUE)
                                 
                                 # Check for failure
                                 if (inherits(m, "try-error")) {
                                   tol <- tol * 10
                                   
                                   # Fallback to e1071 if kernlab keeps failing
                                   if (tol > 1) {
                                     self$method <- "e1071"
                                     self$model <- e1071::svm(
                                       x = x_mat, 
                                       y = y, 
                                       kernel = "linear", 
                                       probability = TRUE,
                                       ...
                                     )
                                     break
                                   }
                                 } else {
                                   # Check if kernlab's probability model is sane
                                   if (all(sapply(m@prob.model, function(mm) length(mm) == 2))) {
                                     self$model <- m
                                     break
                                   } else {
                                     # If prob model is weird, treat as failure and relax tol
                                     tol <- tol * 10
                                   }
                                 }
                               }
                               
                               invisible(self)
                             },
                             
                             predict = function(x) {
                               if (is.null(self$model)) stop("Model not trained.")
                               x_mat <- as.matrix(x)
                               
                               if (self$method == "kernlab") {
                                 # --- kernlab prediction ---
                                 prob <- predict(self$model, newdata = x_mat, type = "probabilities")
                                 # Ensure column order matches our class names
                                 colnames(prob) <- self$class_names
                                 
                               } else {
                                 # --- e1071 prediction ---
                                 pred_obj <- predict(self$model, newdata = x_mat, probability = TRUE)
                                 prob <- attr(pred_obj, "probabilities")
                                 # e1071 might swap column order; reorder to match self$class_names
                                 prob <- prob[, self$class_names, drop = FALSE]
                               }
                               
                               # Determine class based on highest probability
                               pred_idx <- max.col(prob)
                               pred <- factor(self$class_names[pred_idx], levels = self$class_names)
                               
                               return(list(pred = pred, prob = prob))
                             }
                           )
)
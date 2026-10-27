library(R6)
library(ranger)

classifier_RF <- R6Class("classifier_RF",
                         public = list(
                           model = NULL,
                           var_names = NULL,
                           class_names = NULL,
                           
                           fit = function(x, y, num.trees = 500, mtry_frac = NULL, ...) {
                             # ranger prefers data.frame or matrix, but x is usually a matrix/df already
                             self$var_names <- colnames(x)
                             self$class_names <- levels(y)
                             
                             # Train using ranger
                             # probability = TRUE allows for class probability estimation
                             args <- list(x = x, y = y, num.trees = num.trees, probability = TRUE, ...)
                             if (!is.null(mtry_frac)) args$mtry <- min(ncol(x), max(1L, round(mtry_frac * ncol(x))))
                             self$model <- do.call(ranger::ranger, args)
                             
                             invisible(self)
                           },
                           
                           predict = function(x) {
                             if (is.null(self$model)) stop("Model not trained.")
                             
                             x_df <- as.data.frame(x)
                             if (!is.null(self$var_names)) colnames(x_df) <- self$var_names
                             
                             # ranger prediction returns a list; we need $predictions for the matrix
                             # type = "response" with a probability model returns probabilities
                             pred_obj <- predict(
                               object = self$model,
                               data = x_df
                             )
                             
                             prob <- pred_obj$predictions
                             colnames(prob) <- self$class_names
                             
                             # Determine class based on max probability
                             pred_idx <- max.col(prob)
                             pred <- factor(
                               self$class_names[pred_idx], 
                               levels = self$class_names
                             )
                             
                             return(list(pred = pred, prob = prob))
                           }
                         )
)
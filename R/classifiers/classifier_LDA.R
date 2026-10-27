library(R6)
library(MASS)

classifier_LDA <- R6Class("classifier_LDA",
                          public = list(
                            model = NULL,
                            class_names = NULL,
                            
                            fit = function(x, y, ...) {
                              self$class_names <- levels(y)
                              
                              # MASS::lda uses a matrix or data frame
                              # ... allows for 'prior' or 'method' arguments
                              self$model <- MASS::lda(x = as.matrix(x), grouping = y, ...)
                              
                              invisible(self)
                            },
                            
                            predict = function(x) {
                              if (is.null(self$model)) stop("Model not trained.")
                              
                              # MASS predict returns a list with $class and $posterior
                              p <- predict(self$model, newdata = as.matrix(x))
                              
                              # Standardize output
                              prob <- p$posterior
                              colnames(prob) <- self$class_names
                              
                              # Ensure factor levels are consistent
                              pred <- factor(p$class, levels = self$class_names)
                              
                              return(list(pred = pred, prob = prob))
                            }
                          )
)
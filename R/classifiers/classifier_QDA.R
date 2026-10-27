classifier_QDA <- R6Class("classifier_QDA",
                          public = list(
                            model = NULL,
                            class_names = NULL,
                            
                            fit = function(x, y, ...) {
                              self$class_names <- levels(y)
                              self$model <- MASS::qda(x = as.matrix(x), grouping = y, ...)
                              invisible(self)
                            },
                            
                            predict = function(x) {
                              p <- predict(self$model, newdata = as.matrix(x))
                              prob <- p$posterior
                              colnames(prob) <- self$class_names
                              return(list(pred = p$class, prob = prob))
                            }
                          )
)
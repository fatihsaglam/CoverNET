library(R6)
library(FNN)

classifier_KNN <- R6Class("classifier_KNN",
                          public = list(
                            train_x = NULL,
                            train_y = NULL,
                            var_names = NULL,
                            class_names = NULL,
                            k = NULL,
                            
                            fit = function(x, y, k = 5, ...) {
                              # FNN is "lazy", so we store the training data
                              self$train_x <- as.matrix(x)
                              self$train_y <- y
                              self$var_names <- colnames(x)
                              self$class_names <- levels(y)
                              self$k <- as.integer(k)
                              
                              invisible(self)
                            },
                            
                            predict = function(x) {
                              if (is.null(self$train_x)) stop("Model not trained.")
                              
                              test_x <- as.matrix(x)
                              if (!is.null(self$var_names)) colnames(test_x) <- self$var_names
                              
                              neighbor_indices <- FNN::get.knnx(
                                data = self$train_x, 
                                query = test_x, 
                                k = self$k + 1)$nn.index[,-1]
                              
                              prob <- t(apply(
                                neighbor_indices,
                                1,
                                \(row) {
                                  freq <- table(self$train_y[row])
                                  c(freq) / self$k
                                }
                              ))
                              colnames(prob) <- self$class_names
                              pred <- factor(self$class_names[max.col(prob)], self$class_names)

                              return(list(pred = pred, prob = prob))
                            }
                          )
)

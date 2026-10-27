library(R6)
library(xgboost)

classifier_GBM <- R6Class("classifier_GBM",
                          public = list(
                            model = NULL,
                            var_names = NULL,
                            class_names = NULL,
                            
                            fit = function(x, y, nrounds = 100, eta = 0.3, max_depth = 6, ...) {
                              self$var_names <- colnames(x)
                              self$class_names <- levels(y)
                              
                              # XGBoost needs 0-indexed numeric labels: [0, 1]
                              y_num <- as.integer(y) - 1L
                              
                              # Create DMatrix
                              dtrain <- xgboost::xgb.DMatrix(data = as.matrix(x), label = y_num)
                              
                              # Default parameters for binary classification
                              params <- list(
                                objective = "binary:logistic",
                                eval_metric = "logloss",
                                eta = eta,
                                max_depth = max_depth
                              )
                              
                              # Train the model
                              self$model <- xgboost::xgb.train(
                                params = params, 
                                data = dtrain, 
                                nrounds = nrounds, 
                                ...
                              )
                              
                              invisible(self)
                            },
                            
                            predict = function(x) {
                              if (is.null(self$model)) stop("Model not trained.")
                              
                              # Convert input to matrix for DMatrix
                              x_mat <- as.matrix(x)
                              if (!is.null(self$var_names)) colnames(x_mat) <- self$var_names
                              
                              # Predict returns probabilities for the '1' class
                              prob_pos <- predict(self$model, newdata = x_mat)
                              
                              # Construct probability matrix [Class1, Class2]
                              prob <- cbind(1 - prob_pos, prob_pos)
                              colnames(prob) <- self$class_names
                              
                              # Determine class based on 0.5 threshold/max probability
                              pred_idx <- max.col(prob)
                              pred <- factor(self$class_names[pred_idx], levels = self$class_names)
                              
                              return(list(pred = pred, prob = prob))
                            }
                          )
)
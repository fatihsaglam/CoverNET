library(R6)

classifier_LogReg <- R6Class("LogReg",
                        public = list(
                          model = NULL,
                          var_names = NULL,
                          class_names = NULL,
                          
                          # 'fit' updates the object in-place using binomial GLM
                          fit = function(x, y) {
                            # Ensure data is in data.frame format for glm
                            x_df <- as.data.frame(x)
                            self$var_names <- colnames(x_df)
                            self$class_names <- levels(y)
                            
                            # Check for binary classification
                            if (length(self$class_names) != 2L) {
                              stop("Logistic Regression (binomial) requires exactly 2 classes.")
                            }
                            
                            # Combine into a single data frame for the formula interface
                            dat <- data.frame(x_df, y = y)
                            
                            # Train using standard Logistic Regression
                            self$model <- glm(
                              formula = y ~ .,
                              data = dat,
                              family = "binomial",
                              control = glm.control(maxit = 1000)
                            )
                            
                            invisible(self)
                          },
                          
                          # 'predict' calculates probabilities and maps to factors
                          predict = function(x) {
                            if (is.null(self$model)) {
                              stop("Model must be fit before calling predict.")
                            }
                            
                            x_df <- as.data.frame(x)
                            colnames(x_df) <- self$var_names
                            
                            # 'response' type gives P(y = second_level)
                            prob_pos <- predict(self$model, newdata = x_df, type = "response")
                            
                            # Construct probability matrix [Class1, Class2]
                            prob_mat <- cbind(1 - prob_pos, prob_pos)
                            colnames(prob_mat) <- self$class_names
                            
                            # Determine class based on 0.5 threshold
                            # Using levels directly to ensure factor consistency
                            pred_labels <- ifelse(prob_pos < 0.5, self$class_names[1], self$class_names[2])
                            pred <- factor(pred_labels, levels = self$class_names)
                            
                            return(list(pred = pred, prob = prob_mat))
                          }
                        )
)

# --- Usage ---
# my_lr <- LogReg_Class$new()
# my_lr$fit(X_train, y_train)
# results <- my_lr$predict(X_test)
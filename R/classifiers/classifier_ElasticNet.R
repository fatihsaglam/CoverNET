library(R6)
library(glmnet)

classifier_ElasticNet <- R6Class("classifier_ElasticNet",
                                 public = list(
                                   model = NULL,
                                   class_names = NULL,
                                   var_names = NULL,
                                   
                                   fit = function(x, y, alpha = 0.5, ...) {
                                     self$class_names <- levels(y)
                                     self$var_names <- colnames(x)
                                     
                                     # glmnet requires matrix input and numeric y
                                     # alpha=1 is Lasso, alpha=0 is Ridge, alpha=0.5 is Elastic Net
                                     self$model <- glmnet::cv.glmnet(
                                       x = as.matrix(x), 
                                       y = y, 
                                       family = "binomial", 
                                       alpha = alpha,
                                       ...
                                     )
                                     invisible(self)
                                   },
                                   
                                   predict = function(x) {
                                     # Use lambda.1se for a simpler, more robust model
                                     prob_pos <- predict(self$model, newx = as.matrix(x), s = "lambda.1se", type = "response")
                                     prob <- cbind(1 - prob_pos, prob_pos)
                                     colnames(prob) <- self$class_names
                                     
                                     pred_idx <- max.col(prob)
                                     pred <- factor(self$class_names[pred_idx], levels = self$class_names)
                                     
                                     return(list(pred = pred, prob = prob))
                                   }
                                 )
)
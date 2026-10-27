library(R6)
library(ada)

classifier_AdaBoost <- R6Class("classifier_AdaBoost",
                               public = list(
                                 model = NULL,
                                 class_names = NULL,
                                 var_names = NULL,
                                 
                                 fit = function(x, y, iter = 50, ...) {
                                   self$class_names <- levels(y)
                                   self$var_names <- colnames(x)
                                   
                                   # ada::ada usually takes a formula or x,y separately
                                   # iter corresponds to the number of boosting iterations
                                   self$model <- ada::ada(
                                     x = as.data.frame(x), 
                                     y = y, 
                                     iter = iter, 
                                     ...
                                   )
                                   
                                   invisible(self)
                                 },
                                 
                                 predict = function(x) {
                                   if (is.null(self$model)) stop("Model not trained.")
                                   
                                   x_df <- as.data.frame(x)
                                   colnames(x_df) <- self$var_names
                                   
                                   # Get probabilities
                                   # ada returns a matrix where columns are the classes
                                   prob <- predict(self$model, newdata = x_df, type = "probs")
                                   colnames(prob) <- self$class_names
                                   
                                   # Get hard class predictions
                                   p_class <- predict(self$model, newdata = x_df, type = "vector")
                                   
                                   # Ensure factor levels are consistent with training
                                   pred <- factor(p_class, levels = self$class_names)
                                   
                                   return(list(pred = pred, prob = prob))
                                 }
                               )
)
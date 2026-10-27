library(R6)
library(rpart)

classifier_CART <- R6Class("classifier_CART",
                           public = list(
                             model = NULL,
                             var_names = NULL,
                             class_names = NULL,
                             
                             fit = function(x, y, ...) {
                               # Prepare data
                               x_df <- as.data.frame(x)
                               self$var_names <- colnames(x_df)
                               self$class_names <- levels(y)
                               
                               # Combine into data frame for formula interface
                               dat <- data.frame(x_df, y = y)
                               
                               # Train rpart model
                               # ... allows for control params like cp, minsplit, etc.
                               self$model <- rpart::rpart(
                                 formula = y ~ ., 
                                 data = dat, 
                                 method = "class", 
                                 ...
                               )
                               
                               invisible(self)
                             },
                             
                             predict = function(x) {
                               if (is.null(self$model)) stop("Model not trained.")
                               
                               # Ensure data matches training features
                               x_df <- as.data.frame(x)
                               colnames(x_df) <- self$var_names
                               
                               # Get class labels
                               pred <- predict(
                                 object = self$model, 
                                 newdata = x_df, 
                                 type = "class"
                               )
                               
                               # Get probability matrix
                               prob <- predict(
                                 object = self$model, 
                                 newdata = x_df, 
                                 type = "prob"
                               )
                               
                               # Ensure factor levels and column names match original y
                               colnames(prob) <- self$class_names
                               pred <- factor(pred, levels = self$class_names)
                               
                               return(list(pred = pred, prob = prob))
                             }
                           )
)
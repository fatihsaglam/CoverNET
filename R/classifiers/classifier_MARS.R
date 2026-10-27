library(R6)
library(earth)

classifier_MARS <- R6Class("classifier_MARS",
                           public = list(
                             model = NULL,
                             var_names = NULL,
                             class_names = NULL,
                             
                             fit = function(x, y, nk = 100, nprune = 100, degree = 1, ...) {
                               self$var_names <- colnames(x)
                               self$class_names <- levels(y)
                               
                               # Train MARS with a binomial GLM wrapper
                               self$model <- earth::earth(
                                 x = x,
                                 y = y,
                                 nk = nk,
                                 nprune = nprune,
                                 degree = degree,
                                 glm = list(family = "binomial"),
                                 ...
                               )
                               
                               invisible(self)
                             },
                             
                             predict = function(x) {
                               if (is.null(self$model)) stop("Model not trained.")
                               
                               x_df <- as.data.frame(x)
                               if (!is.null(self$var_names)) colnames(x_df) <- self$var_names
                               
                               # Get class predictions
                               pred_raw <- predict(self$model, newdata = x_df, type = "class")
                               pred <- factor(pred_raw, levels = self$class_names)
                               
                               # Get response probabilities (P of the second level/positive class)
                               prob_pos <- predict(self$model, newdata = x_df, type = "link") # type="link" for GLM portion
                               # Converting from logit to probability if necessary, 
                               # but earth with type="response" is usually cleaner:
                               prob_pos <- predict(self$model, newdata = x_df, type = "response")
                               
                               # Construct the probability matrix [Class1, Class2]
                               # earth's 'response' for binomial returns the probability of the SUCCESS class
                               prob_mat <- cbind(1 - prob_pos, prob_pos)
                               colnames(prob_mat) <- self$class_names
                               
                               return(list(pred = pred, prob = prob_mat))
                             }
                           )
)
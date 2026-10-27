library(R6)
library(BART)

classifier_BART <- R6Class("classifier_BART",
                           public = list(
                             model = NULL,
                             var_names = NULL,
                             class_names = NULL,
                             
                             fit = function(x, y, ndpost = 100, ntree = 20L, ...) {
                               self$var_names <- colnames(x)
                               self$class_names <- levels(y)
                               
                               # Train the model once. 
                               # We use mbart2 but we don't provide x.test here.
                               # This generates the posterior of the tree structures.
                               self$model <- BART::mbart2(
                                 x.train = as.matrix(x),
                                 y.train = as.integer(y),
                                 ndpost = ndpost,
                                 ntree = ntree,
                                 silent = TRUE,
                                 ...
                               )
                               
                               invisible(self)
                             },
                             
                             predict = function(x) {
                               if (is.null(self$model)) stop("Model not trained.")
                               
                               x_mat <- as.matrix(x)
                               if (!is.null(self$var_names)) colnames(x_mat) <- self$var_names
                               
                               # Use the S3 predict method for bart objects.
                               # This uses the previously stored trees to calculate 
                               # probabilities for new data WITHOUT re-training.
                               pred_obj <- predict(self$model, newdata = x_mat)
                               
                               # BART returns probabilities in $prob.test (if x.test was provided)
                               # or we calculate the mean from the posterior draws.
                               # pred_obj is usually a matrix [draws, observations * classes]
                               
                               k_class <- length(self$class_names)
                               n_obs <- nrow(x_mat)
                               
                               # Flattened posterior means for each class
                               # BART's predict output for multiclass can be a bit complex; 
                               # we colMeans the draws to get the average probability per observation.
                               prob_means <- colMeans(pred_obj)
                               
                               # Reshape into [n_obs, k_class]
                               prob <- matrix(
                                 prob_means,
                                 nrow = n_obs,
                                 ncol = k_class,
                                 byrow = TRUE
                               )
                               
                               colnames(prob) <- self$class_names
                               
                               # Determine Class
                               pred_idx <- max.col(prob)
                               pred <- factor(self$class_names[pred_idx], levels = self$class_names)
                               
                               return(list(pred = pred, prob = prob))
                             }
                           )
)

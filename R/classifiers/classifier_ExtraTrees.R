classifier_ExtraTrees <- R6Class("classifier_ExtraTrees",
                                 public = list(
                                   model = NULL,
                                   class_names = NULL,
                                   var_names = NULL,
                                   
                                   fit = function(x, y, ...) {
                                     self$class_names <- levels(y)
                                     self$var_names <- colnames(x)
                                     # splitrule = "extratrees" makes it an Extra Trees model
                                     self$model <- ranger::ranger(
                                       x = x, y = y, 
                                       probability = TRUE, 
                                       splitrule = "extratrees", 
                                       ...
                                     )
                                     invisible(self)
                                   },
                                   
                                   predict = function(x) {
                                     p <- predict(self$model, data = as.data.frame(x))$predictions
                                     colnames(p) <- self$class_names
                                     pred <- factor(self$class_names[max.col(p)], levels = self$class_names)
                                     return(list(pred = pred, prob = p))
                                   }
                                 )
)
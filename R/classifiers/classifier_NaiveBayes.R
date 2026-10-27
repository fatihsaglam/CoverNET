library(R6)
library(naivebayes)

classifier_NaiveBayes <- R6Class(
  "NB",
  public = list(
    model = NULL,
    class_names = NULL,
    var_names = NULL,

    # 'fit' updates the object in-place
    fit = function(x, y, ...) {
      # Ensure data is in a friendly format
      x_df <- as.data.frame(x)
      self$var_names <- colnames(x_df)
      self$class_names <- levels(y)

      # Train the model using the naivebayes package
      # The '...' allows passing laplace, usekernel, etc.
      self$model <- naivebayes::naive_bayes(x = x_df, y = y, ...)

      # Return self invisibly to allow for method chaining
      invisible(self)
    },

    # 'predict' uses the stored model
    predict = function(x) {
      if (is.null(self$model)) {
        stop("The model has not been trained yet. Call $fit() first.")
      }

      x_df <- as.data.frame(x)

      # Ensure column names match training data
      if (!is.null(self$var_names)) {
        colnames(x_df) <- self$var_names
      }

      # Generate Class Predictions
      pred <- predict(object = self$model, newdata = x_df, type = "class")

      # Generate Probabilities
      prob <- predict(object = self$model, newdata = x_df, type = "prob")

      return(list(pred = pred, prob = prob))
    }
  )
)

# --- Usage Example ---
# my_nb <- NB_Class$new()
# my_nb$fit(iris[,1:4], iris$Species, laplace = 1)
# results <- my_nb$predict(iris[1:5, 1:4])
library(R6)
library(reticulate)

classifier_KANNET <- R6Class(
  "KANNET",
  public = list(
    model = NULL,
    device = NULL,
    class_names = NULL,
    var_names = NULL,
    feature_importance = NULL,
    
    initialize = function() {
      torch <- import("torch")
      self$device <- if (torch$cuda$is_available()) {
        torch$device("cuda")
      } else {
        torch$device("cpu")
      }
    },
    
    fit = function(
    x,
    y,
    n_grid = 20L,
    k = 3L,                   # Order of the spline (3 is standard cubic)
    hidden_dim = 20L,
    n_hidden = 1L,
    activation = "silu",      # Base function for KAN
    epochs = 100L,
    batch_size = 64L,
    lr = 1e-3
    ) {
      torch <- import("torch")
      nn <- import("torch.nn", convert = TRUE)
      kan <- import("kan", convert = TRUE) # Requires the pykan python package
      
      # Data prep
      x <- as.matrix(x)
      self$var_names <- colnames(x)
      self$class_names <- levels(y)
      k_class <- length(self$class_names)
      
      # PyTorch CrossEntropyLoss expects 0-indexed long tensors
      y_num <- as.integer(y) - 1L
      
      # Map string to base activation function
      act_fn <- switch(tolower(activation),
                       "silu"    = nn$SiLU(),
                       "relu"    = nn$ReLU(),
                       "tanh"    = nn$Tanh(),
                       "sigmoid" = nn$Sigmoid(),
                       nn$SiLU())
      
      # KAN Network Width: [input_dim, hidden_dim_1, ..., output_dim]
      width_list <- as.integer(c(ncol(x), rep(hidden_dim, n_hidden), k_class))
      
      # Initialize KAN
      self$model <- kan$KAN(
        width = width_list,
        grid = as.integer(n_grid),
        k = as.integer(k),
        base_fun = act_fn
      )
      
      self$model$to(self$device)
      
      optim <- torch$optim$RMSprop(self$model$parameters(), lr = lr)
      crit <- nn$CrossEntropyLoss()
      
      x_t <- torch$tensor(x, dtype = torch$float32)
      y_t <- torch$tensor(y_num, dtype = torch$long) # CrossEntropy needs Long types
      
      # Training Loop with Batching
      for (epoch in seq_len(epochs)) {
        idx <- sample.int(nrow(x))
        batches <- split(idx, ceiling(seq_along(idx) / batch_size))
        
        for (ids in batches) {
          xb <- x_t[ids - 1, ]$to(self$device)
          yb <- y_t[ids - 1, ]$to(self$device)
          
          optim$zero_grad()
          logits <- self$model(xb)
          loss <- crit(logits, yb)
          loss$backward()
          optim$step()
        }
      }
      
      # Extract Feature Importance (if supported by the pykan version)
      if (!is.null(self$model$feature_score)) {
        self$feature_importance <- self$model$feature_score$detach()$cpu()$numpy()
      }
      
      invisible(self)
    },
    
    predict = function(x) {
      if (is.null(self$model)) stop("Model not trained.")
      
      torch <- import("torch")
      x_t <- torch$tensor(as.matrix(x), dtype = torch$float32)$to(self$device)
      
      with(torch$no_grad(), {
        logits <- self$model(x_t)
        # Convert logits to probabilities across the class dimension (dim=1)
        probs_tensor <- torch$softmax(logits, dim = 1L)
        prob <- probs_tensor$cpu()$numpy()
      })
      
      # Format Output
      colnames(prob) <- self$class_names
      pred_idx <- max.col(prob)
      pred <- factor(self$class_names[pred_idx], levels = self$class_names)
      
      return(list(pred = pred, prob = prob))
    }
  )
)
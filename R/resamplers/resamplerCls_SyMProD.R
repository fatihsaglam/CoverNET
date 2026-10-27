resamplerCls_SyMProD <- function(x, y, NT = 3, CT = 0, K = 5, M = 5, ovRate = 1, ...) {
  x <- as.matrix(x)
  y <- as.factor(y)
  var_names <- colnames(x)
  p <- ncol(x)
  
  maj_class <- names(which.max(table(y)))
  min_class <- names(which.min(table(y)))
  
  x_min <- x[y == min_class, , drop = FALSE]
  x_maj <- x[y == maj_class, , drop = FALSE]
  
  n_min <- nrow(x_min)
  n_maj <- nrow(x_maj)
  ngen <- (n_maj - n_min)*ovRate
  
  # Step 1: Noise Removal (Z-score filtering)
  z_min <- scale(x_min)
  z_maj <- scale(x_maj)
  
  x_min_clean <- x_min[apply(abs(z_min) <= NT, 1, all), , drop = FALSE]
  x_maj_clean <- x_maj[apply(abs(z_maj) <= NT, 1, all), , drop = FALSE]
  
  if (nrow(x_maj_clean) == 0 |nrow(x_min_clean) == 0) {
    return(list(
      x_new = x,
      y_new = y,
      x_syn = data.frame(matrix(NA, nrow = 0, ncol = ncol(x))),
      y_syn = factor(c(), levels = levels(y))
    ))
  }
  
  # Step 2: Compute closeness factor for each minority instance
  dist_matrix <- as.matrix(dist(x_min_clean))
  total_dists <- rowSums(dist_matrix)
  C_min <- 1 / total_dists
  C_min <- scales::rescale(C_min)
  if (any(is.nan(C_min))) {
    C_min <- rep(1e-100,length(C_min))
  }
  
  # Step 3: Filter based on tau_min vs tau_maj and calculate probabilities
  prob_scores <- numeric(nrow(x_min_clean))
  keep_flags <- logical(nrow(x_min_clean))
  
  KK <- min(K, nrow(x_min_clean) - 1, nrow(x_maj_clean) - 1)
  idx_min_nn <- FNN::get.knnx(data = x_min_clean, query = x_min_clean, k = KK + 1)$nn.index[, -1, drop = FALSE]
  idx_maj_nn <- FNN::get.knnx(data = x_maj_clean, query = x_min_clean, k = KK)$nn.index
  
  for (i in seq_len(nrow(x_min_clean))) {
    x_i <- x_min_clean[i, , drop = FALSE]

    idx_min <- idx_min_nn[i,,drop = FALSE]
    idx_maj <- idx_maj_nn[i,,drop = FALSE]
    
    tau_min <- sum(C_min[idx_min] / sqrt(rowSums((x_min_clean[idx_min,, drop = FALSE] - x_i[rep(1, KK),, drop = FALSE])^2) + 1e-10))
    dists <- as.matrix(dist(rbind(
      x_i,
      x_maj_clean[idx_maj, , drop = FALSE]
    )))[,1]
    tau_maj <- sum(1 / dists) 
    if (is.infinite(tau_maj)) {
      tau_maj <- 1e100
    }
    
    if (tau_min > CT * tau_maj) {
      phi <- (tau_min + 1) / (tau_maj + 1)
      prob_scores[i] <- phi
      keep_flags[i] <- TRUE
    }
  }
  
  prob_scores <- prob_scores[keep_flags]
  prob_scores <- prob_scores / sum(prob_scores)
  x_min_filtered <- x_min_clean[keep_flags, , drop = FALSE]
  
  # Step 4: Generate synthetic samples
  # x_syn <- matrix(NA, nrow = ngen, ncol = p)
  x_syn <- matrix(NA, nrow = 0, ncol = p)
  
  if (nrow(x_min_filtered) >= 1) {
    while (nrow(x_syn) < ngen) {
      idx_base <- sample(1:nrow(x_min_filtered), size = 1, prob = prob_scores)
      x_base <- x_min_filtered[idx_base, , drop = FALSE]
      
      MM <- min(M, nrow(x_min_filtered) - 1)
      if (MM < 0) {
        break
      }
      idx_nn <- FNN::get.knnx(x_min_filtered, x_base, k = MM + 1)$nn.index[1, ]
      R <- x_min_filtered[idx_nn, , drop = FALSE]
      beta <- runif(MM + 1)
      beta <- beta / sum(beta)
      
      x_syn <- rbind(x_syn, colSums(sweep(R, 1, beta, FUN = "*")))
    }
  }

  # for (i in 1:ngen) {
  #   idx_base <- sample(1:nrow(x_min_filtered), size = 1, prob = prob_scores)
  #   x_base <- x_min_filtered[idx_base, , drop = FALSE]
  #   
  #   MM <- min(M, nrow(x_min_filtered) - 1)
  #   idx_nn <- FNN::get.knnx(x_min_filtered, x_base, k = MM + 1)$nn.index[1, ]
  #   R <- x_min_filtered[idx_nn, , drop = FALSE]
  #   beta <- runif(MM + 1)
  #   beta <- beta / sum(beta)
  #   
  #   # Assuming uniform Pr weights as in original paper
  #   x_syn[i, ] <- colSums(sweep(R, 1, beta, FUN = "*"))
  # }
  
  x_new <- rbind(x, x_syn)
  y_new <- factor(c(as.character(y), rep(min_class, nrow(x_syn))), levels = levels(y))
  colnames(x_new) <- var_names
  
  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = factor(rep(min_class, nrow(x_syn)), levels = levels(y))
  ))
}

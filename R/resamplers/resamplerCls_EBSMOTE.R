resamplerCls_EBSMOTE <- function(x, y,
                                 m_alpha = 5,
                                 m_beta = 5,
                                 lambda = 3,
                                 v_alpha = 5,
                                 w_alpha = 5,
                                 expand_alpha = 0.3,
                                 ovRate = 1,
                                 ...) {
  n_needed = NULL
  x <- as.matrix(x)
  y <- as.factor(y)
  p <- ncol(x)
  colnames(x) <- colnames(x)
  
  maj_class <- names(which.max(table(y)))
  min_class <- names(which.min(table(y)))
  
  x_min <- x[y == min_class, , drop = FALSE]
  x_maj <- x[y == maj_class, , drop = FALSE]
  n_min <- nrow(x_min)
  n_maj <- nrow(x_maj)
  
  n_needed <- (n_maj - n_min)*ovRate
  
  mm_alpha <- min(m_alpha, nrow(x) - 1)
  n_alpha <- sapply(1:n_min, function(i) {
    knn_indices <- FNN::get.knnx(x, x_min[i,,drop=FALSE], k = mm_alpha)$nn.index[1,]
    sum(y[knn_indices] == maj_class)
  })
  
  border_idx <- which(n_alpha >= mm_alpha / lambda & n_alpha < mm_alpha)
  x_min_border <- x_min[border_idx,,drop=FALSE]
  
  n_maj <- nrow(x_maj)
  mm_beta <- min(m_beta, nrow(x) - 1)
  n_beta <- sapply(1:n_maj, function(i) {
    knn_indices <- FNN::get.knnx(x, x_maj[i,,drop = FALSE], k = mm_beta)$nn.index[1,]
    sum(y[knn_indices] == min_class)
  })
  
  border_idx_maj <- which(n_beta >= mm_beta / lambda & n_beta < mm_beta)
  x_maj_border <- x_maj[border_idx_maj,,drop = FALSE]
  
  x_syn <- matrix(NA, nrow = 0, ncol = p)

  while (nrow(x_syn) < n_needed) {
    if (nrow(x_maj_border) < 1) {
      break
    }
    if (nrow(x_min_border) < 1) {
      break
    }
    
    pi_idx <- sample(1:nrow(x_min_border), 1)
    pi <- x_min_border[pi_idx,, drop = FALSE]
    vv_alpha <- min(v_alpha, nrow(x_maj_border))
    t_neighbors_idx <- FNN::get.knnx(x_maj_border, pi, k = vv_alpha)$nn.index[1,]
    tj <- x_maj_border[sample(t_neighbors_idx, 1), , drop = FALSE]
    
    ww_alpha <- min(w_alpha, nrow(x_min_border))
    pk_neighbors_idx <- FNN::get.knnx(x_min_border, tj, k = ww_alpha)$nn.index[1,]
    pk <- x_min_border[sample(pk_neighbors_idx, 1), , drop = FALSE]
    
    phi <- runif(1)
    gnew <- (1 - expand_alpha) * (phi * pi + (1 - phi) * pk) + expand_alpha * tj
    x_syn <- rbind(x_syn, gnew)
  }
  
  x_new <- rbind(x, x_syn)
  y_new <- factor(c(as.character(y), rep(min_class, nrow(x_syn))), levels = levels(y))
  
  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = factor(rep(min_class, nrow(x_syn)), levels = levels(y))
  ))
}

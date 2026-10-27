# x <- imbalance::banana[, 1:2]
# y <- imbalance::banana[, 3]
# 
# ovRate <- 1
# k <- 5

resamplerCls_BiSMOTE <- function(x, y, k = 5, ovRate = 1, ...) {
  
  x <- as.matrix(x)
  p <- ncol(x)
  n <- nrow(x)
  class_names <- levels(y)
  n_classes <- sapply(class_names, function(m)
    sum(y == m))
  k_class <- length(class_names)
  
  n_needed <- ovRate * (max(n_classes) - n_classes)
  x_syn_list <- lapply(1:k_class, function(m)
    matrix(NA, nrow = 0, ncol = p))
  
  for (i_class in 1:k_class) {
    if (n_needed[[i_class]] == 0) {
      next
    }
    
    x_main <- x[y == class_names[i_class], , drop = FALSE]
    x_other <- x[y != class_names[i_class], , drop = FALSE]
    n_main <- nrow(x_main)
    n_other <- nrow(x_other)
    
    m_main2other <- FNN::get.knnx(data = x_other, query = x_main, k = 1)
    NN_main2other <- m_main2other$nn.index
    d_main2other <- m_main2other$nn.dist
    
    m_main2main <- FNN::get.knnx(data = x_main,
                                 query = x_main,
                                 k = k + 1)
    NN_main2main <- m_main2main$nn.index[, -1, drop = FALSE]
    d_main2main <- m_main2main$nn.dist[, -1, drop = FALSE]
    
    n_syn <- 0
    while (n_syn < n_needed[i_class]) {
      i_center <- sample(1:n_main, size = 1)
      x_center <- x_main[i_center, , drop = FALSE]
      i_selectedNeighbor <- sample(1:k, 1)
      i_target <- NN_main2main[i_center, i_selectedNeighbor]
      x_target <- x_main[i_target, , drop = FALSE]
      
      d_limit <- d_main2other[i_center,]
      d_center2target <- d_main2main[i_center, i_selectedNeighbor]
      if (d_center2target < 1e-100) {
        d_center2target <- 1e-100
      }
      lambda <- runif(1, -d_limit/d_center2target, d_limit/d_center2target)
      
      x_syn_step <- x_center + (x_target - x_center)*lambda
      x_syn_list[[i_class]] <- rbind(x_syn_list[[i_class]], x_syn_step)
      n_syn <- nrow(x_syn_list[[i_class]])
    }
  }
  
  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(unlist(sapply(1:k_class, function(m) {
    rep(class_names[m], nrow(x_syn_list[[m]]))
  })), levels = class_names)
  
  x_new <- rbind(x, x_syn)
  y_new <- c(y, y_syn)
  
  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn
  ))
}

# 
# m_BiSMOTE <- resamplerCls_BiSMOTE(x = x, y = y)
# plot(x, col = y)
# points(x_syn, col = "green")


resamplerCls_LRDO <- function(x, y, k = 5, type = "type 2", ovRate = 1, giveEllipses = FALSE, ...) {
  covComed <- function (X, n.iter = 2, reweight = FALSE, tolSolve = control$tolSolve, 
                        trace = control$trace, wgtFUN = control$wgtFUN, control = robustbase:::rrcov.control()) 
  {
    
    COM <- function (X) 
    {
      stopifnot(robustbase:::is.1num(p <- ncol(X)), p >= 1)
      med <- robustbase::colMedians(X)
      Y <- sweep(X, 2L, med, `-`)
      COM <- matrix(0, p, p)
      madY <- numeric(p)
      for (i in 1:p) {
        madY[i] <- madYi <- mad(Yi <- Y[, i]) + 1e-5
        for (j in seq_len(i - 1)) {
          COM[j, i] <- COM[i, j] <- median(Yi * Y[, j])/(madYi * 
                                                           madY[j])
        }
        COM[i, i] <- median(Yi^2)/(madYi^2)
      }
      1.4826^2 * COM
    }
    
    comedian <- function (rho, Z, X) 
    {
      p <- ncol(X)
      U <- svd(rho, nv = 0L)$u
      madX <-apply(X, 2, function(m) mad(m) + 1e-10)
      I.mad <- 1/madX
      Q <- madX * U
      t.inv.Q <- I.mad * U
      Z <- X %*% t.inv.Q
      madZ <- apply(Z, 2, mad)
      list(Q = Q, Z = Z, S. = tcrossprod(Q * rep(madZ, each = p)))
    }
    
    mahalanobis <- function (x, center, cov, inverted = FALSE, ...) 
    {
      x <- if (is.vector(x)) 
        matrix(x, ncol = length(x))
      else as.matrix(x)
      if (!isFALSE(center)) 
        x <- sweep(x, 2L, center)
      if (!inverted) 
        cov <- MASS::ginv(cov, ...)
      setNames(rowSums(x %*% cov * x), rownames(x))
    }
    
    d <- dim(X <- as.matrix(X))
    n <- d[1]
    p <- d[2]
    if (is.character(wgtFUN)) 
      wgtFUN <- robustbase:::.wgtFUN.covComed[[wgtFUN]](p = p, n = n, control)
    if (!is.function(wgtFUN)) 
      stop("'wgtFUN' must be a function or a string specifying such a function")
    madX <- apply(X, 2, function(m) mad(m) + 1e-10)
    I.mad <- 1/madX
    rho <- I.mad * COM(X) * rep(I.mad, each = p)
    U <- svd(rho, p, nv = 0L)$u
    t.inv.Q <- I.mad * U
    Z <- X %*% t.inv.Q
    out <- comedian(rho, Z, X)
    for (it in seq_len(n.iter)) out <- comedian(out$S., out$Z, 
                                                X)
    mm <- robustbase::colMedians(out$Z)
    mx <- drop(out$Q %*% mm)
    mah <- mahalanobis(X, mx, out$S., tol = tolSolve)
    weights <- wgtFUN(mah)
    covW <- cov.wt(X, wt = weights)[c("cov", "center", "n.obs")]
    covW$weights <- if (reweight) {
      covW$raw.weights <- weights
      covW$mah <- mahalanobis(X, covW$center, covW$cov, tol = tolSolve)
      wgtFUN(mah)
    }
    else weights
    structure(class = "comed", c(list(Z = out$Z, raw.cov = out$S., 
                                      raw.center = mx, raw.mah = mah, wgtFUN = wgtFUN), covW))
  }
  
  p <- ncol(x)
  n <- nrow(x)
  var_names <- colnames(x)
  x <- as.matrix(x)
  
  class_names <- levels(y)
  k_class <- length(class_names)
  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])
  n_classes <- sapply(x_classes, nrow)
  
  # dens_classes <- lapply(1:k_class, function(m) {
  #   H <- ks::Hns(x_classes[[m]])
  #   ks::kde(x_classes[[m]], H = H, eval.points = x_classes[[m]])$estimate
  # })

  # dens_classes <- lapply(1:k_class, function(m) {
  #   m_mclust <- Mclust(data = x_classes[[m]], modelNames = "VVV", verbose = FALSE)
  #   dens(data = x_classes[[m]], modelName = m_mclust$modelName, parameters = m_mclust$parameters)
  # })
  
  covMu_classes <- lapply(x_classes, function(m) cov(m)/nrow(m))
  
  n_needed <- ceiling((max(n_classes) - n_classes)*ovRate)
  x_syn_list <- list()
  for (i in 1:k_class) {
    if (n_needed[[i]] == 0) {
      next
    }
    
    x_syn_list[[i]] <- matrix(data = NA, nrow = 0, ncol = p)
    NN_main2main <- FNN::get.knnx(data = x_classes[[i]], query = x_classes[[i]], k = k + 1)$nn.index[,-1, drop = FALSE]
    
    while (nrow(x_syn_list[[i]]) < n_needed[i]) {
      # i_selected <- sample(1:n_classes[i], size = 1, prob = dens_classes[[i]])
      i_selected <- sample(1:n_classes[i], size = 1)
      
      if (type == "type 1") {
        x_local <- x_classes[[i]][NN_main2main[i_selected,],, drop = FALSE]
        Sigma <- cov(x_local)
        mu <- colMeans(x_local)
        
        x_syn_step <- mvtnorm::rmvnorm(n = 1, mean = mu, sigma = Sigma)
        x_syn_list[[i]] <- rbind(x_syn_list[[i]], x_syn_step)
      }
      
      if (type == "type 2") {
        x_local <- x_classes[[i]][NN_main2main[i_selected,],, drop = FALSE]
        m_robust <- try(covComed(x_local), silent = TRUE)
        
        if (inherits(m_robust, "try-error")) {
          Sigma <- cov(x_local)
          mu <- colMeans(x_local)
        } else {
          Sigma <- m_robust$cov
          mu <- m_robust$center
        }
        
        x_syn_step <- mvtnorm::rmvnorm(n = 1, mean = mu, sigma = Sigma)
        x_syn_list[[i]] <- rbind(x_syn_list[[i]], x_syn_step)
      }
      
      if (type == "type 3") {
        Sigma <- covMu_classes[[i]]
        mu <- x_classes[[i]][i_selected,]
        
        x_syn_step <- mvtnorm::rmvnorm(n = 1, mean = mu, sigma = Sigma)
        x_syn_list[[i]] <- rbind(x_syn_list[[i]], x_syn_step)
      }
    }
  }
  
  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(unlist(sapply(1:k_class, function(m) rep(class_names[m], n_needed[m]))), levels = class_names, labels = class_names)
  
  x_new <- rbind(x, x_syn)
  y_new <- c(y, y_syn)
  
  colnames(x_new) <- var_names
  
  if (giveEllipses) {
    for (i in 1:k_class) {
      
      NN_main2main <- FNN::get.knnx(data = x_classes[[i]], query = x_classes[[i]], k = k + 1)$nn.index[,-1, drop = FALSE]
      Ellipses <- list()
      
      for (j in 1:n_classes[i]) {
        x_local <- x_classes[[i]][NN_main2main[j,],, drop = FALSE]
        m_robust <- try(covComed(x_local), silent = TRUE)
        
        if (inherits(m_robust, "try-error")) {
          Sigma <- cov(x_local)
          mu <- colMeans(x_local)
        } else {
          Sigma <- m_robust$cov
          mu <- m_robust$center
        }
        
        Ellipses[[j]] <- list(
          Sigma = Sigma,
          mu = mu
        )
      }
    }
  } else {
    Ellipses <- NULL
  }
  
  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn,
    Ellipses = Ellipses
  ))
}

# x <- imbalance::banana[,1:2]
# y <- imbalance::banana[,3]
# 
# m_LRMV <- resamplerCls_LRMV(x = x, y = y, type = "type 3")
# 
# plot(x, col = y)
# points(m_LRMV$x_syn, col = "green")

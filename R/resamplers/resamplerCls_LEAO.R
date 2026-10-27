# Local Entropy-Adversarial Oversampling (LEAO)
# Implements Algorithm 1 + Eqs. (1)–(6) in the LEAO paper. :contentReference[oaicite:0]{index=0}
#
# Output contract (matches your resampler API):
# - x_new, y_new : full resampled training set
# - x_syn, y_syn : only synthetic samples (minority label)

resamplerCls_LEAO <- function(
    x, y,
    # LEAO hyperparameters (paper-recommended defaults: iters=50, alpha=1, k=3, stop_prob=0.3)
    iters = 50,
    alpha = 1,
    prob_type = c("bagging", "logistic", "knn", "nb"),
    k = 3,
    stop_prob = 0.3,
    # step size control (paper: random per synthetic instance)
    step_max = 0.10,
    step_min = NULL,
    # numeric stability
    eps = 1e-6,
    cov_ridge = 1e-8,
    seed = NULL,
    ...
) {
  if (!is.null(seed)) set.seed(seed)
  
  # ---------- input checks ----------
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) stop("x must be a matrix or data.frame.")
  if (!is.numeric(x)) stop("x must be numeric.")
  n <- nrow(x)
  p <- ncol(x)
  if (length(y) != n) stop("y length must equal nrow(x).")
  
  if (!is.factor(y)) y <- factor(y)
  if (nlevels(y) != 2) stop("LEAO implementation here is for binary classification (2 classes).")
  
  tab <- table(y)
  minority_label <- names(tab)[which.min(tab)]
  majority_label <- names(tab)[which.max(tab)]
  
  idx_min <- which(y == minority_label)
  idx_maj <- which(y == majority_label)
  
  n_min <- length(idx_min)
  n_maj <- length(idx_maj)
  
  n_needed <- n_maj - n_min
  if (n_needed <= 0) {
    return(list(
      x_new = x, y_new = y,
      x_syn = x[0, , drop = FALSE],
      y_syn = y[0]
    ))
  }
  
  k <- as.integer(k)
  if (k < 1) stop("k must be >= 1.")
  k <- min(k, n_maj) # cap
  
  prob_type <- match.arg(prob_type)
  if (!is.null(step_min)) {
    if (step_min <= 0 || step_min >= step_max) stop("step_min must be in (0, step_max).")
  } else {
    step_min <- step_max / 10
  }
  
  # ---------- helpers ----------
  # stable entropy: standard binary entropy (positive). Paper formula shows p log p + ...; we use the standard -sum.
  entropy_bin <- function(pv) {
    pv <- pmin(pmax(pv, 0), 1)
    pv1 <- pv + eps
    pv0 <- (1 - pv) + eps
    - (pv1 * log(pv1) + pv0 * log(pv0))
  }
  
  # Mahalanobis distance using global covariance (paper uses S from Xmaj ∪ Xmin).
  S <- stats::cov(x)
  S <- S + diag(cov_ridge, p)
  Sinv <- tryCatch(solve(S), error = function(e) MASS::ginv(S))
  
  dM <- function(xa, xb) {
    v <- xa - xb
    as.numeric(sqrt(drop(t(v) %*% Sinv %*% v)))
  }
  
  # gradient of L(x) (Eq. 3). returns vector length p
  grad_L <- function(x, x_min, x_maj, H_min, H_maj) {
    dm_min <- dM(x, x_min)
    dm_maj <- dM(x, x_maj)
    
    # avoid division by zero
    dm_min <- max(dm_min, 1e-12)
    dm_maj <- max(dm_maj, 1e-12)
    
    term_min <- (-alpha) * H_min * exp(-alpha * dm_min) * (Sinv %*% (x - x_min)) / dm_min
    term_maj <- ( alpha) * H_maj * exp(-alpha * dm_maj) * (Sinv %*% (x - x_maj)) / dm_maj
    as.numeric(term_min + term_maj)
  }
  
  L_value <- function(x, x_min, x_maj, H_min, H_maj) {
    H_min * exp(-alpha * dM(x, x_min)) - H_maj * exp(-alpha * dM(x, x_maj))
  }
  
  # ---------- probability estimation p(x) for entropy ----------
  # returns p(minority | x) for every row in x
  estimate_p_minority <- function(X, y_fac, type) {
    y01 <- as.integer(y_fac == minority_label)
    
    if (type == "logistic") {
      dat <- data.frame(y = y01, X)
      fit <- suppressWarnings(stats::glm(y ~ ., data = dat, family = stats::binomial()))
      p_hat <- suppressWarnings(stats::predict(fit, newdata = dat, type = "response"))
      return(pmin(pmax(p_hat, 0), 1))
    }
    
    if (type == "nb") {
      if (!requireNamespace("e1071", quietly = TRUE)) {
        warning("Package 'e1071' not available; falling back to logistic for prob_type='nb'.")
        return(estimate_p_minority(X, y_fac, "logistic"))
      }
      dat <- data.frame(y = y_fac, X)
      fit <- e1071::naiveBayes(y ~ ., data = dat)
      pr <- predict(fit, newdata = dat, type = "raw")
      # probability of minority class
      return(pmin(pmax(pr[, minority_label], 0), 1))
    }
    
    if (type == "knn") {
      if (!requireNamespace("FNN", quietly = TRUE)) {
        warning("Package 'FNN' not available; falling back to logistic for prob_type='knn'.")
        return(estimate_p_minority(X, y_fac, "logistic"))
      }
      # Using leave-one-out via knn (approx): get kNN among the full set excluding self by using k+1 and dropping self if present.
      kk <- min(11, nrow(X) - 1) # reasonable default if paper doesn't fix it
      nn <- FNN::get.knn(X, k = kk)
      # majority vote probability among neighbors (approx)
      p_hat <- vapply(seq_len(nrow(X)), function(i) {
        neigh <- nn$nn.index[i, ]
        mean(y01[neigh])
      }, numeric(1))
      return(pmin(pmax(p_hat, 0), 1))
    }
    
    # bagging (paper recommended)
    if (type == "bagging") {
      # Prefer ipred::bagging; fallback to randomForest; fallback to logistic
      if (requireNamespace("ipred", quietly = TRUE) && requireNamespace("rpart", quietly = TRUE)) {
        dat <- data.frame(y = y_fac, X)
        fit <- ipred::bagging(y ~ ., data = dat, nbagg = 50, coob = TRUE)
        pr <- predict(fit, newdata = dat, type = "prob")
        return(pmin(pmax(pr[, minority_label], 0), 1))
      }
      if (requireNamespace("randomForest", quietly = TRUE)) {
        dat <- data.frame(y = y_fac, X)
        fit <- randomForest::randomForest(y ~ ., data = dat, ntree = 200)
        pr <- predict(fit, newdata = dat, type = "prob")
        return(pmin(pmax(pr[, minority_label], 0), 1))
      }
      warning("Bagging backend not available (ipred/rpart or randomForest). Falling back to logistic.")
      return(estimate_p_minority(X, y_fac, "logistic"))
    }
    
    stop("Unknown prob_type.")
  }
  
  # ---------- compute entropy for all instances (minority + majority) ----------
  p_hat <- estimate_p_minority(x, y, prob_type)
  H_all <- entropy_bin(p_hat)
  
  # paper uses entropy of xmin and xmaj in L(x)
  H_min <- H_all[idx_min]
  H_maj <- H_all[idx_maj]
  
  # ---------- entropy-weighted allocation of synthetic counts to minority seeds (Eq. 5–6) ----------
  w <- H_min
  if (all(w <= 1e-12)) {
    w <- rep(1, length(w))
  }
  w <- w / sum(w)
  
  # allocate integer counts summing to n_needed
  # use multinomial allocation for exact sum
  counts_per_seed <- as.integer(rmultinom(1, size = n_needed, prob = w))
  # mapping to original row indices
  seeds <- idx_min
  
  # ---------- precompute k nearest majority neighbors for each minority seed ----------
  Xmaj <- x[idx_maj, , drop = FALSE]
  Xmin <- x[idx_min, , drop = FALSE]
  
  # compute neighbor indices in majority set using Mahalanobis distance
  get_k_nearest_majority_idx <- function(xm) {
    # distances to all majority
    diffs <- sweep(Xmaj, 2, xm, "-")
    # compute quadratic form efficiently: rowSums((diffs %*% Sinv) * diffs)
    qf <- rowSums((diffs %*% Sinv) * diffs)
    ord <- order(qf, decreasing = FALSE)
    ord[seq_len(k)]
  }
  
  maj_neighbors_list <- lapply(seq_len(n_min), function(i) get_k_nearest_majority_idx(Xmin[i, ]))
  
  # ---------- synthesis ----------
  x_syn <- matrix(NA_real_, nrow = n_needed, ncol = p)
  colnames(x_syn) <- colnames(x)
  y_syn <- factor(rep(minority_label, n_needed), levels = levels(y))
  
  row_out <- 1L
  
  for (i in seq_len(n_min)) {
    ni <- counts_per_seed[i]
    if (ni <= 0) next
    
    x_min_i <- Xmin[i, ]
    H_min_i <- H_min[i]
    neigh_ids <- maj_neighbors_list[[i]]
    
    for (j in seq_len(ni)) {
      # select one majority neighbor uniformly among k nearest (Algorithm 1 line 13)
      pick <- sample(neigh_ids, size = 1)
      x_maj_j <- Xmaj[pick, ]
      H_maj_j <- H_maj[which(idx_maj == idx_maj[pick])[1]] # safe mapping
      
      # init synthetic position (Algorithm 1 line 16)
      x_cur <- as.numeric(x_min_i)
      
      # random step size per synthetic instance (paper: varies across instances)
      eta <- stats::runif(1, min = step_min, max = step_max)
      
      # gradient descent (Eqs. 3–4) + probabilistic termination
      for (t in seq_len(iters)) {
        # early termination with probability stop_prob (Algorithm 1 line 19)
        if (stats::runif(1) < stop_prob) break
        
        g <- grad_L(x_cur, x_min_i, x_maj_j, H_min_i, H_maj_j)
        x_next <- x_cur - eta * g
        
        # accept if reached near-zero / non-positive adversarial potential (Algorithm 1 lines 21–24)
        if (L_value(x_next, x_min_i, x_maj_j, H_min_i, H_maj_j) <= 0) {
          x_cur <- x_next
          break
        }
        
        x_cur <- x_next
      }
      
      x_syn[row_out, ] <- x_cur
      row_out <- row_out + 1L
    }
  }
  
  # in rare numeric/alloc edge cases, row_out might not fill perfectly; trim NAs
  if (row_out <= n_needed) {
    keep <- which(!is.na(x_syn[, 1]))
    x_syn <- x_syn[keep, , drop = FALSE]
    y_syn <- y_syn[keep]
  }
  
  x_new <- rbind(x, x_syn)
  y_new <- factor(c(as.character(y), as.character(y_syn)), levels = levels(y))
  
  list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn
  )
}

# 
# data <- datasets_binary$banana
# x <- data[,1:2]
# y <- data[,3]
# 
# m_LEAO <- resamplerCls_LEAO(
#   x = x, y = y,
#   iters = 50,
#   alpha = 1,
#   # prob_type = c("bagging", "logistic", "knn", "nb"),
#   prob_type = c("nb"),
#   k = 3,
#   stop_prob = 0.3,
#   # step size control (paper: random per synthetic instance)
#   step_max = 0.10,
#   step_min = NULL,
#   # numeric stability
#   eps = 1e-6,
#   cov_ridge = 1e-8,
#   seed = NULL
#   )
# 
# plot(x, col = y)
# points(m_LEAO$x_syn, col = "green")

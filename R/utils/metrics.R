metricCls_ACC <- function(truth, pred, prob = NULL, ...) {
  mean(truth == pred)
}

metricCls_BACC <- function(truth, pred, prob = NULL, ...) {
  class_names <- levels(truth)
  k_classes <- length(class_names)
  n <- length(truth)
  n_classes <- sapply(class_names, function(m) {
    sum(truth == m)
  })
  mean(sapply(class_names, function(m) {
    ii <- truth == m
    sum(truth[ii] == pred[ii])
  })/n_classes)
}

metricCls_MCC <- function(truth, pred, prob = NULL, ...) {
  class_names <- levels(truth)
  k_classes <- length(class_names)
  n <- length(truth)
  n_classes <- sapply(class_names, function(m) {
    sum(truth == m)
  })
  mean(sapply(class_names, function(m) {
    truth_temp <- as.character(truth)
    pred_temp <- as.character(pred)
    
    truth_temp[!(truth == m)] <- "rrest"
    pred_temp[!(pred_temp == m)] <- "rrest"
    
    truth_temp <- factor(truth_temp, levels = c(m, "rrest"))
    pred_temp <- factor(pred_temp, levels = c(m, "rrest"))
    
    mlr::measureMCC(truth = truth_temp, response = pred_temp, negative = "rrest", positive = m)
  }))
}

metricCls_F1 <- function(truth, pred, prob = NULL, ...) {
  class_names <- levels(truth)
  k_classes <- length(class_names)
  n <- length(truth)
  n_classes <- sapply(class_names, function(m) {
    sum(truth == m)
  })
  mean(sapply(class_names, function(m) {
    truth_temp <- as.character(truth)
    pred_temp <- as.character(pred)
    
    truth_temp[!(truth == m)] <- "rrest"
    pred_temp[!(pred_temp == m)] <- "rrest"
    
    truth_temp <- factor(truth_temp, levels = c(m, "rrest"))
    pred_temp <- factor(pred_temp, levels = c(m, "rrest"))
    
    mlr::measureF1(truth = truth_temp, response = pred_temp, positive = m)
  }))
}

metricCls_GMEAN <- function(truth, pred, prob = NULL, ...) {
  
  class_names <- levels(truth)
  k_classes <- length(class_names)
  n <- length(truth)
  n_classes <- sapply(class_names, function(m) {
    sum(truth == m)
  })
  mean(sapply(class_names, function(m) {
    truth_temp <- as.character(truth)
    pred_temp <- as.character(pred)
    
    truth_temp[!(truth == m)] <- "rrrest"
    pred_temp[!(pred_temp == m)] <- "rrrest"
    
    truth_temp <- factor(truth_temp, levels = c(m, "rrrest"))
    pred_temp <- factor(pred_temp, levels = c(m, "rrrest"))
    
    mlr::measureGMEAN(truth = truth_temp, response = pred_temp, negative = "rrrest", positive = m)
  }))
}

metricCls_ROCAUC <- function(truth, pred = NULL, prob = NULL, ...) {
  if (is.null(prob)) {
    if (is.null(pred)) {
      stop("Either pred or prob must be provied")
    }
    
    class_names <- levels(truth)
    k_class <- length(class_names)
    prob <- matrix(data = 0, nrow = length(pred), ncol = k_class)
    pred_num <- as.numeric(pred)
    for (i in 1:length(pred)) {
      prob[i,pred_num[i]] <- 1
    }
  }
  yardstick::roc_auc_vec(truth = truth, estimate = prob, estimator = "macro")
}

metricCls_PRAUC <- function(truth, pred = NULL, prob = NULL, ...) {
  if (is.null(prob)) {
    if (is.null(pred)) {
      stop("Either pred or prob must be provied")
    }
    
    class_names <- levels(truth)
    k_class <- length(class_names)
    prob <- matrix(data = 0, nrow = length(pred), ncol = k_class)
    pred_num <- as.numeric(pred)
    for (i in 1:length(pred)) {
      prob[i,pred_num[i]] <- 1
    }
  }
  yardstick::pr_auc_vec(truth = truth, estimate = prob, estimator = "macro")
}

metricCls_IAM <- function(truth, pred, prob = NULL, ...) {
  class_names <- levels(truth)
  k_classes <- length(class_names)
  C <- table(truth, pred)
  
  mean(sapply(1:k_classes, function(m) {
    (C[m,m] - max(sum(C[-m,m]),sum(C[m,-m])))/max(sum(C[m,]), sum(C[,m]))
  }))
}

metricCls_Brier <- function(truth, pred = NULL, prob, ...) {
  truth_num <- as.numeric(truth) - 1
  BrierScore <- sum(sapply(1:ncol(prob), function(m) {
    sum((
      (prob[,m] - 1)^2 # 1 comes from true probability p_i
    )[I(truth_num == (m - 1))])
  }))/length(truth)
  return(BrierScore)
}

# Balanced Brier
metricCls_BBrier <- function(truth, pred = NULL, prob, ...) {
  truth_num <- as.numeric(truth) - 1
  BalancedBrierScore <- mean(sapply(1:ncol(prob), function(m) {
    mean((
      (prob[,m] - 1)^2 # 1 comes from true probability p_i
    )[I(truth_num == (m - 1))])
  }))
  return(BalancedBrierScore)
}

# Penalized Brier
metricCls_PBrier <- function(truth, pred = NULL, prob, ...) {
  c <- ncol(prob)  # Number of classes
  n <- nrow(prob)  # Number of samples
  
  pbs <- 0  # Initialize the penalized Brier Score accumulator
  
  for (i in 1:n) {
    true_class <- as.numeric(truth[i])  # Get the true class index
    true_vector <- rep(0, c)
    true_vector[true_class] <- 1  # One-hot encoding of true class
    
    pred_probs <- prob[i, ]  # Predicted probabilities for this instance
    
    # Brier score component
    brier_component <- sum((true_vector - pred_probs)^2)
    
    # Penalization if the predicted class is incorrect
    predicted_class <- which.max(pred_probs)
    penalty <- ifelse(predicted_class != true_class, (c - 1) / c, 0)
    
    pbs <- pbs + brier_component + penalty
  }
  
  return(pbs / n)  # Return the average PBS
}

metricCls_Kappa <- function(truth, pred, prob = NULL, ...) {
  Kappa <- mlr::measureKAPPA(truth = truth, response = pred)
  
  return(Kappa)
}

metricCls_P4 <- function(truth, pred, prob = NULL, class_pos = NULL, class_neg = NULL, ...) {
  if (any(is.null(class_pos), is.null(class_neg))) {
    class_names <- levels(truth)
    class_pos <- names(which.min(table(truth)))
    class_neg <- class_names[class_names != class_pos]
  }
  
  tp <- sum(truth == class_pos & pred == class_pos)
  fp <- sum(truth == class_neg & pred == class_pos)
  tn <- sum(truth == class_neg & pred == class_neg)
  fn <- sum(truth == class_pos & pred == class_neg)
  
  if (tp == 0 & tn == 0) {
    P4 <- 0
  } else {
    P4 <- (4*tp*tn)/(4*tp*tn + (tp + tn)*(fp + fn))
  }
  
  return(P4)
}

metricCls_all <- list(
  ACC = metricCls_ACC,
  BACC = metricCls_BACC,
  MCC = metricCls_MCC,
  F1 = metricCls_F1,
  GMEAN = metricCls_GMEAN,
  ROCAUC = metricCls_ROCAUC,
  PRAUC = metricCls_PRAUC,
  IAM = metricCls_IAM,
  Brier = metricCls_Brier,
  BBrier = metricCls_BBrier,
  PBrier = metricCls_PBrier,
  Kappa = metricCls_Kappa,
  P4 = metricCls_P4
)

#' Helper to list available metrics
list_metrics <- function() {
  names(metricCls_all)
}


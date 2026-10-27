# ==============================================================================
# File: R/classifiers/classifier_CCCDNNET_ablation.R   (NEW FILE - nothing existing is modified)
# Purpose: CoverNET with switchable dominant-point initialisation, explicit seeding,
#          and bookkeeping of epochs / init time / training time.
#
# It re-uses NetCCCD and CCCDLoss from classifier_CCCDNNET.R (same Python session)
# and defines its own training function (abl_train_cccd), so the original
# train_cccd / classifier_CCCDNNET stay untouched.
#
# init = "kmeans" : class-wise k-means on the penultimate representations (paper's method)
#        "sample" : K_c randomly chosen training representations per class (random-but-on-data)
#        "randn"  : the layer's default N(0, 1) draw (no data-dependent init at all)
#
# seed            : seeds torch (hidden weights + default mu draw) and two numpy
#                   RandomState streams (validation split / batch order, and random init
#                   choices). Every variant that gets the same seed therefore sees the SAME
#                   hidden-layer init, validation split and batch order -> paired comparison.
# ==============================================================================

py_run_string("
import time
import numpy as np
import torch
import torch.optim as optim

def abl_train_cccd(x_train, y_train, p, h, n_layers, act_name,
                   epochs, batch_size, lr, k_pos, k_neg,
                   lambda1, lambda2, lambda3, device_name,
                   patience=10, init='kmeans', seed=1):

    from sklearn.cluster import KMeans

    x_train = np.array(x_train, dtype=np.float32)
    y_int   = np.array(y_train, dtype=np.int32).flatten()
    device  = torch.device(device_name)

    seed = int(seed)
    torch.manual_seed(seed)
    rng_data = np.random.RandomState(seed)           # validation split + batch order
    rng_init = np.random.RandomState(seed + 10000)   # random initialisation choices

    model = NetCCCD(p, h, n_layers, act_name,
                    k_pos=int(k_pos), k_neg=int(k_neg)).to(device)

    # ---- stratified validation split (drawn first: identical across variants)
    pos_idx   = np.where(y_int == 1)[0]
    neg_idx   = np.where(y_int == 0)[0]
    n_val_pos = max(1, int(len(pos_idx) * 0.1))
    n_val_neg = max(1, int(len(neg_idx) * 0.1))
    val_idx   = np.concatenate([
        rng_data.choice(pos_idx, n_val_pos, replace=False),
        rng_data.choice(neg_idx, n_val_neg, replace=False)
    ])
    train_idx = np.setdiff1d(np.arange(len(x_train)), val_idx)

    # ---- initialisation of the dominant points
    t_init = time.time()
    if init != 'randn':
        with torch.no_grad():
            h_all = model.hidden(
                torch.tensor(x_train, dtype=torch.float32).to(device)).cpu().numpy()

        def get_centers(hh, k):
            k = max(1, min(int(k), len(hh)))
            if init == 'kmeans':
                if k == 1:
                    return hh[rng_init.choice(len(hh), 1)]
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(hh)
                return km.cluster_centers_
            if init == 'sample':
                return hh[rng_init.choice(len(hh), k, replace=False)]
            raise ValueError('unknown init: ' + str(init))

        centers_pos = get_centers(h_all[y_int == 1], k_pos)
        centers_neg = get_centers(h_all[y_int == 0], k_neg)
        with torch.no_grad():
            model.cccd.mu_pos.copy_(torch.tensor(centers_pos, dtype=torch.float32).to(device))
            model.cccd.mu_neg.copy_(torch.tensor(centers_neg, dtype=torch.float32).to(device))
    init_seconds = time.time() - t_init

    # ---- data tensors
    x_tr = torch.tensor(x_train[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_int[train_idx],   dtype=torch.float32).reshape(-1, 1)
    x_vl = torch.tensor(x_train[val_idx],   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_int[val_idx],     dtype=torch.float32).reshape(-1, 1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CCCDLoss(lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    # ---- training loop with early stopping (same rule as the original)
    best_val_loss  = float('inf')
    best_state     = None
    best_epoch     = 0
    patience_count = 0
    epochs_run     = 0
    n_tr           = len(train_idx)
    t_train        = time.time()
    model.train()

    for epoch in range(int(epochs)):
        order = rng_data.permutation(n_tr)
        for i in range(0, n_tr, int(batch_size)):
            ids = order[i : i + int(batch_size)]
            xb  = x_tr[ids].to(device)
            yb  = y_tr[ids].to(device)
            optimizer.zero_grad()
            p_pos, p_neg, s_pos, s_neg = model(xb)
            loss = criterion(p_pos, p_neg, s_pos, s_neg, yb, model.cccd)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p_pos_v, p_neg_v, s_pos_v, s_neg_v = model(x_vl)
            val_loss = criterion(p_pos_v, p_neg_v, s_pos_v, s_neg_v,
                                 y_vl, model.cccd).item()
        model.train()
        epochs_run = epoch + 1

        if val_loss < best_val_loss - 1e-4:
            best_val_loss  = val_loss
            best_epoch     = epoch + 1
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= int(patience):
                break

    train_seconds = time.time() - t_train

    if best_state is not None:
        model.load_state_dict(best_state)

    model.n_epochs      = int(epochs_run)
    model.best_epoch    = int(best_epoch)
    model.init_seconds  = float(init_seconds)
    model.train_seconds = float(train_seconds)

    flush_memory()
    return model
")


classifier_CCCDNNET_abl <- R6Class(
  "classifier_CCCDNNET_abl",
  public = list(
    model         = NULL,
    device        = "cpu",
    class_names   = NULL,
    class_pos     = NULL,
    n_epochs      = NA_integer_,
    best_epoch    = NA_integer_,
    init_seconds  = NA_real_,
    train_seconds = NA_real_,

    initialize = function() {
      torch <- import("torch")
      self$device <- if (torch$cuda$is_available()) "cuda" else "cpu"
    },

    fit = function(x, y,
                   hidden_dim = 32L,
                   n_hidden   = 2L,
                   activation = "relu",
                   epochs     = 100L,
                   batch_size = 256L,
                   lr         = 1e-3,
                   k_pos      = NULL,
                   k_neg      = NULL,
                   lambda1    = 1.0,
                   lambda2    = 1.0,
                   lambda3    = 0.1,
                   patience   = 10L,
                   init       = "kmeans",
                   seed       = 1L) {

      x_mat <- as.matrix(x)
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)

      # Same default rule as the paper: K_c = max(1, floor(sqrt(n_c)))
      n_pos <- sum(y_bin == 1)
      n_neg <- sum(y_bin == 0)
      if (is.null(k_pos)) k_pos <- max(1L, floor(sqrt(n_pos)))
      if (is.null(k_neg)) k_neg <- max(1L, floor(sqrt(n_neg)))

      self$model <- py$abl_train_cccd(
        x_train     = x_mat,
        y_train     = y_bin,
        p           = ncol(x_mat),
        h           = as.integer(hidden_dim),
        n_layers    = as.integer(n_hidden),
        act_name    = activation,
        epochs      = as.integer(epochs),
        batch_size  = as.integer(batch_size),
        lr          = lr,
        k_pos       = as.integer(k_pos),
        k_neg       = as.integer(k_neg),
        lambda1     = lambda1,
        lambda2     = lambda2,
        lambda3     = lambda3,
        device_name = self$device,
        patience    = as.integer(patience),
        init        = init,
        seed        = as.integer(seed)
      )

      self$n_epochs      <- as.integer(self$model$n_epochs)
      self$best_epoch    <- as.integer(self$model$best_epoch)
      self$init_seconds  <- as.numeric(self$model$init_seconds)
      self$train_seconds <- as.numeric(self$model$train_seconds)
      invisible(self)
    },

    # identical to classifier_CCCDNNET$predict (uses the original predict_cccd)
    predict = function(x) {
      probs_pos <- py$predict_cccd(self$model, as.matrix(x), self$device)
      prob <- matrix(NA, nrow = length(probs_pos), ncol = 2)
      prob[, which(self$class_names == self$class_pos)] <- probs_pos
      prob[, which(self$class_names != self$class_pos)] <- 1 - probs_pos
      colnames(prob) <- self$class_names
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      list(pred = pred, prob = prob)
    }
  )
)

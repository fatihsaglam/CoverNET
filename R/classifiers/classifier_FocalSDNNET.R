py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FocalLossSD(nn.Module):
    '''
    Sample-dependent focal loss (FLSD) from Mukhoti et al. (2020).
    gamma is chosen per sample based on predicted probability for
    the ground truth class, following the FLSD-53 schedule:
      gamma = 5 if p_hat < threshold
      gamma = 3 if p_hat >= threshold
    This encourages higher entropy predictions and better calibration.
    '''
    def __init__(self, gamma_low=5.0, gamma_high=3.0, threshold=0.2):
        super(FocalLossSD, self).__init__()
        self.gamma_low  = gamma_low
        self.gamma_high = gamma_high
        self.threshold  = threshold

    def forward(self, logits, targets):
        probs  = torch.sigmoid(logits)                          # (N, 1)
        pt     = torch.where(targets == 1, probs, 1 - probs)   # p_hat for ground truth class
        
        # Sample-dependent gamma selection
        gamma  = torch.where(pt < self.threshold,
                             torch.full_like(pt, self.gamma_low),
                             torch.full_like(pt, self.gamma_high))
        
        loss = -((1 - pt) ** gamma) * torch.log(pt + 1e-8)
        return loss.mean()

def train_focal_sd(x_train, y_train, p, h, n_layers, act_name,
                   epochs, batch_size, lr,
                   gamma_low, gamma_high, threshold,
                   device_name, patience=10):

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32).flatten()

    device    = torch.device(device_name)
    model     = Net(p, h, n_layers, act_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLossSD(gamma_low=gamma_low,
                            gamma_high=gamma_high,
                            threshold=threshold)

    # --- Stratified validation split ---
    pos_idx   = np.where(y_train == 1)[0]
    neg_idx   = np.where(y_train == 0)[0]
    n_val_pos = max(1, int(len(pos_idx) * 0.1))
    n_val_neg = max(1, int(len(neg_idx) * 0.1))
    val_idx   = np.concatenate([
        np.random.choice(pos_idx, n_val_pos, replace=False),
        np.random.choice(neg_idx, n_val_neg, replace=False)
    ])
    train_idx = np.setdiff1d(np.arange(len(x_train)), val_idx)

    x_t  = torch.tensor(x_train[train_idx], dtype=torch.float32)
    y_t  = torch.tensor(y_train[train_idx], dtype=torch.float32).reshape(-1, 1)
    x_vt = torch.tensor(x_train[val_idx],   dtype=torch.float32).to(device)
    y_vt = torch.tensor(y_train[val_idx],   dtype=torch.float32).reshape(-1, 1).to(device)

    # --- Early stopping state ---
    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0
    patience       = patience

    n_tr = len(train_idx)
    model.train()

    for epoch in range(int(epochs)):
        indices = np.random.permutation(n_tr)
        for i in range(0, n_tr, int(batch_size)):
            ids = indices[i : i + int(batch_size)]
            xb, yb = x_t[ids].to(device), y_t[ids].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_vt), y_vt).item()
        model.train()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    flush_memory()
    return model
")

classifier_FocalSDNNET <- R6Class(
  "classifier_FocalSDNNET",
  public = list(
    model       = NULL,
    device      = "cpu",
    class_names = NULL,
    class_pos   = NULL,
    
    initialize = function() {
      torch <- import("torch")
      self$device <- if (torch$cuda$is_available()) "cuda" else "cpu"
    },
    
    fit = function(x, y, hidden_dim = 32L, n_hidden = 2L,
                   activation = "relu", epochs = 50L,
                   batch_size = 64L, lr = 1e-3,
                   gamma_low = 5.0, gamma_high = 3.0,
                   threshold = 0.2, patience = 10) {
      
      x_mat <- as.matrix(x)
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)
      
      self$model <- py$train_focal_sd(
        x_train    = x_mat,
        y_train    = y_bin,
        p          = ncol(x_mat),
        h          = hidden_dim,
        n_layers   = n_hidden,
        act_name   = activation,
        epochs     = epochs,
        batch_size = batch_size,
        lr         = lr,
        gamma_low  = gamma_low,
        gamma_high = gamma_high,
        threshold  = threshold,
        device_name = self$device,
        patience   = patience
      )
      invisible(self)
    },
    
    predict = function(x) {
      probs_pos <- py$predict(self$model, as.matrix(x), self$device)
      prob <- matrix(NA, nrow = length(probs_pos), ncol = 2)
      prob[, which(self$class_names == self$class_pos)] <- probs_pos
      prob[, which(self$class_names != self$class_pos)] <- 1 - probs_pos
      colnames(prob) <- self$class_names
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      list(pred = pred, prob = prob)
    }
  )
)
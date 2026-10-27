py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LDAMLoss(nn.Module):
    '''
    Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019).
    Encourages larger margins for minority classes.
    Margin for class j: delta_j = C / n_j^(1/4)
    Binary case: minority=1 (pos), majority=0 (neg)
    '''
    def __init__(self, n_pos, n_neg, C=1.0, weight=None):
        super(LDAMLoss, self).__init__()
        # Compute per-class margins
        delta_pos = C / (n_pos ** 0.25)   # larger margin for minority
        delta_neg = C / (n_neg ** 0.25)   # smaller margin for majority
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.weight    = weight            # optional class reweighting

    def forward(self, logits, targets):
        # logits: (N, 1), targets: (N, 1) binary {0, 1}
        logits  = logits.squeeze(1)        # (N,)
        targets = targets.squeeze(1)       # (N,)

        # Shift logits by class-dependent margin
        # For positive (minority) samples: subtract delta_pos from logit
        # For negative (majority) samples: add delta_neg to logit of wrong class
        margins = torch.where(targets == 1,
                              torch.full_like(logits, self.delta_pos),
                              torch.full_like(logits, -self.delta_neg))
        logits_adjusted = logits - margins

        # Binary cross-entropy on adjusted logits
        if self.weight is not None:
            w = torch.where(targets == 1,
                            torch.full_like(logits, self.weight),
                            torch.ones_like(logits))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits_adjusted, targets, weight=w)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits_adjusted, targets)

        return loss

def train_ldam(x_train, y_train, p, h, n_layers, act_name,
               epochs, batch_size, lr, C, reweight, device_name, patience=10):

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32).flatten()

    device    = torch.device(device_name)
    model     = Net(p, h, n_layers, act_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_pos  = float(np.sum(y_train == 1))
    n_neg  = float(np.sum(y_train == 0))
    weight = (n_neg / n_pos) if reweight else None
    criterion = LDAMLoss(n_pos=n_pos, n_neg=n_neg, C=C, weight=weight)

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

classifier_LDAMNNET <- R6Class(
  "classifier_LDAMNNET",
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
                   C = 1.0, reweight = FALSE, patience = 10) {
      
      x_mat <- as.matrix(x)
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)
      
      self$model <- py$train_ldam(
        x_train     = x_mat,
        y_train     = y_bin,
        p           = ncol(x_mat),
        h           = hidden_dim,
        n_layers    = n_hidden,
        act_name    = activation,
        epochs      = epochs,
        batch_size  = batch_size,
        lr          = lr,
        C           = C,
        reweight    = reweight,
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
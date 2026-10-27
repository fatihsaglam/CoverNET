library(R6)
library(reticulate)

# 1. DEFINE THE PYTHON CLASS IN PURE PYTHON
# This completely bypasses R's 'self' vs Python 'self' conflict.
py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. The Model Architecture
class Net(nn.Module):
    def __init__(self, p, h, n_layers, act_name):
        super(Net, self).__init__()
        act_dict = {'relu': nn.ReLU, 'lrelu': lambda: nn.LeakyReLU(0.2), 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        act_fn = act_dict.get(act_name.lower(), nn.ReLU)
        
        layers = []
        in_f = int(p)
        h_f = int(h)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_f, h_f))
            layers.append(act_fn())
            in_f = h_f
        layers.append(nn.Linear(in_f, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 2. The Full Training Process (The 'Engine')
def train(x_train, y_train, p, h, n_layers, act_name, epochs, batch_size, lr, device_name, pos_weight=None, patience=10):

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32).flatten()

    device    = torch.device(device_name)
    model     = Net(p, h, n_layers, act_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if pos_weight is not None:
        pw        = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

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

# 3. The Prediction Process
def predict(model, x_test, device_name):
    device = torch.device(device_name)
    model.eval()
    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs.flatten()
")

# 2. THE R6 WRAPPER
classifier_NNET <- R6Class(
  "classifier_NNET",
  public = list(
    model = NULL,
    device = "cpu",
    class_names = NULL,
    class_pos = NULL,
    
    initialize = function() {
      torch <- import("torch")
      self$device <- if (torch$cuda$is_available()) "cuda" else "cpu"
    },
    
    fit = function(x, y, hidden_dim = 32L, n_hidden = 2L,
                   activation = "relu", epochs = 50L,
                   batch_size = 64L, lr = 1e-3,
                   balanced = FALSE, patience = 10) {
      
      x_mat <- as.matrix(x)
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)
      
      # Compute pos_weight only if balanced = TRUE
      pw <- if (balanced) {
        n_neg <- sum(y_bin == 0)
        n_pos <- sum(y_bin == 1)
        n_neg / n_pos
      } else {
        NULL
      }
      
      self$model <- py$train(
        x_train     = x_mat,
        y_train     = y_bin,
        p           = ncol(x_mat),
        h           = hidden_dim,
        n_layers    = n_hidden,
        act_name    = activation,
        epochs      = epochs,
        batch_size  = batch_size,
        lr          = lr,
        device_name = self$device,
        pos_weight  = pw,
        patience   = patience
      )
      
      invisible(self)
    },
    
    predict = function(x) {
      probs_pos <- py$predict(self$model, as.matrix(x), self$device)

      # only binary for now
      prob <- matrix(NA, nrow = length(probs_pos), ncol = 2)
      prob[,which(self$class_names == self$class_pos)] <- probs_pos
      prob[,which(self$class_names != self$class_pos)] <- 1 - probs_pos
      colnames(prob) <- self$class_names
      
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      
      return(list(pred = pred, prob = prob))
    }
  )
)
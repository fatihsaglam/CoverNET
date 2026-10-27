# ==============================================================================
# File: R/classifiers/classifier_HeadNNET.R   (NEW FILE, nothing existing is modified)
# Purpose: matched-head comparison (reviewer 1 comment 2, reviewer 3 comment 3). Same hidden layers, same training loop,
#          same BCE loss and early stopping as classifier_NNET; only the output layer differs:
#   head = "mlp" : one extra hidden layer + linear output. Width m is chosen so that the number of head parameters is
#                  about the number of parameters of the CoverNET head, m * (d + 2) + 1 = K * (d + 1), K = k_pos + k_neg.
#   head = "rbf" : standard RBF-network head. K = k_pos + k_neg Gaussian units exp(-||h - mu_k||^2 / r_k^2) with learnable
#                  centres (k-means initialisation per class, as CoverNET) and radii (softplus, initial 0.5), followed by a
#                  learned linear layer (K weights + bias) and sigmoid. No union of balls, no class-wise normalisation.
#   K follows the square-root rule of classifier_CCCDNNET (floor(sqrt(n_c))), unless k_pos / k_neg are given.
# Python names (HeadNet, train_head, predict_head) do not clash with those of the other classifiers.
# ==============================================================================

library(R6)
library(reticulate)

py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HeadNet(nn.Module):
    def __init__(self, p, h, n_layers, act_name, head, k_pos, k_neg):
        super(HeadNet, self).__init__()
        act_dict = {'relu': nn.ReLU, 'lrelu': lambda: nn.LeakyReLU(0.2), 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        act_fn = act_dict.get(act_name.lower(), nn.ReLU)
        layers = []
        in_f = int(p)
        h_f = int(h)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_f, h_f))
            layers.append(act_fn())
            in_f = h_f
        self.hidden = nn.Sequential(*layers)
        self.head = head
        K = int(k_pos) + int(k_neg)
        if head == 'mlp':
            m = max(1, int(round((K * (h_f + 1) - 1) / (h_f + 2))))
            self.out = nn.Sequential(nn.Linear(h_f, m), act_fn(), nn.Linear(m, 1))
        else:
            self.mu = nn.Parameter(torch.randn(K, h_f))
            r_init = torch.log(torch.exp(torch.tensor(0.5)) - 1)
            self.r_raw = nn.Parameter(torch.full((K,), r_init.item()))
            self.w = nn.Linear(K, 1)

    def forward(self, x):
        z = self.hidden(x)
        if self.head == 'mlp':
            return self.out(z)
        sq  = ((z.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2).sum(dim=2)
        r2  = (nn.functional.softplus(self.r_raw) ** 2).unsqueeze(0)
        phi = torch.exp(-sq / (r2 + 1e-8))
        return self.w(phi)

def train_head(x_train, y_train, p, h, n_layers, act_name, epochs, batch_size, lr,
               head, k_pos, k_neg, device_name, patience=10):

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32).flatten()

    device    = torch.device(device_name)
    model     = HeadNet(p, h, n_layers, act_name, head, k_pos, k_neg).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # --- RBF head: centres from k-means on the (initial) hidden representation, per class, as in CoverNET ---
    if head == 'rbf':
        from sklearn.cluster import KMeans
        with torch.no_grad():
            h_all = model.hidden(torch.tensor(x_train, dtype=torch.float32).to(device)).cpu().numpy()
        def get_centers(hh, k):
            k = max(1, min(int(k), len(hh)))
            if k == 1:
                return hh[np.random.choice(len(hh), 1)]
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(hh)
            return km.cluster_centers_
        c_pos = get_centers(h_all[y_train == 1], int(k_pos))
        c_neg = get_centers(h_all[y_train == 0], int(k_neg))
        centers = np.concatenate([c_pos, c_neg], axis=0)
        with torch.no_grad():
            model.mu.data = torch.zeros_like(model.mu.data)
            model.mu.data[:centers.shape[0]] = torch.tensor(centers, dtype=torch.float32).to(device)
            if centers.shape[0] < model.mu.shape[0]:
                model.mu.data[centers.shape[0]:] = torch.tensor(centers[np.random.choice(len(centers), model.mu.shape[0] - centers.shape[0])], dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0
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

def predict_head(model, x_test, device_name):
    device = torch.device(device_name)
    model.eval()
    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x_t)).cpu().numpy()
    return probs.flatten()
")

classifier_HeadNNET <- R6Class(
  "classifier_HeadNNET",
  public = list(
    model       = NULL,
    device      = "cpu",
    class_names = NULL,
    class_pos   = NULL,

    initialize = function() {
      torch <- import("torch")
      self$device <- if (torch$cuda$is_available()) "cuda" else "cpu"
    },

    fit = function(x, y, hidden_dim = 32L, n_hidden = 2L, activation = "relu", epochs = 50L,
                   batch_size = 64L, lr = 1e-3, patience = 10, head = "mlp", k_pos = NULL, k_neg = NULL) {

      x_mat <- as.matrix(x)
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)

      n_pos <- sum(y_bin == 1); n_neg <- sum(y_bin == 0)
      if (is.null(k_pos)) k_pos <- max(1L, floor(sqrt(n_pos)))
      if (is.null(k_neg)) k_neg <- max(1L, floor(sqrt(n_neg)))

      self$model <- py$train_head(
        x_train = x_mat, y_train = y_bin, p = ncol(x_mat), h = hidden_dim, n_layers = n_hidden,
        act_name = activation, epochs = epochs, batch_size = batch_size, lr = lr,
        head = head, k_pos = as.integer(k_pos), k_neg = as.integer(k_neg),
        device_name = self$device, patience = patience)
      invisible(self)
    },

    predict = function(x) {
      probs_pos <- py$predict_head(self$model, as.matrix(x), self$device)
      prob <- matrix(NA, nrow = length(probs_pos), ncol = 2)
      prob[, which(self$class_names == self$class_pos)] <- probs_pos
      prob[, which(self$class_names != self$class_pos)] <- 1 - probs_pos
      colnames(prob) <- self$class_names
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      list(pred = pred, prob = prob)
    }
  )
)

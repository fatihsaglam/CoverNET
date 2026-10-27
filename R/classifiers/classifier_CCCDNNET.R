py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# 1. CCCD FINAL LAYER
# ============================================================

class CCCDLayer(nn.Module):
    '''
    Class Cover Catch Digraph inspired final layer.
    Learns dominant points (prototypes) and radii for each class.
    Uses union-of-balls soft membership for probability computation.
    
    Parameters:
        d        : dimension of penultimate representation
        k_pos    : number of dominant points for minority class
        k_neg    : number of dominant points for majority class
    '''
    def __init__(self, d, k_pos, k_neg):
        super(CCCDLayer, self).__init__()
        
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.d     = d
        
        # Learnable dominant points for each class
        # Shape: (K_c, d)
        self.mu_pos = nn.Parameter(torch.randn(k_pos, d))
        self.mu_neg = nn.Parameter(torch.randn(k_neg, d))
        
        # Learnable raw radii (softplus applied to keep positive)
        # Initialize so that softplus(r_raw) = 0.5
        r_init = torch.log(torch.exp(torch.tensor(0.5)) - 1)  # ~0.541
        self.r_pos_raw = nn.Parameter(torch.full((k_pos,), r_init.item()))
        self.r_neg_raw = nn.Parameter(torch.full((k_neg,), r_init.item()))

    def get_radii(self):
        # Softplus to keep radii positive and differentiable
        r_pos = nn.functional.softplus(self.r_pos_raw)  # (k_pos,)
        r_neg = nn.functional.softplus(self.r_neg_raw)  # (k_neg,)
        return r_pos, r_neg

    def union_of_balls(self, h, mu, r):
        '''
        Compute soft union-of-balls membership.
        h   : (N, d) penultimate representations
        mu  : (K, d) dominant points
        r   : (K,)   radii
        
        Returns: (N,) membership scores in [0, 1]
        S(h) = 1 - prod_k (1 - exp(-||h - mu_k||^2 / r_k^2))
        '''
        # Pairwise squared distances: (N, K)
        # h: (N, d) -> (N, 1, d)
        # mu: (K, d) -> (1, K, d)
        diff    = h.unsqueeze(1) - mu.unsqueeze(0)   # (N, K, d)
        sq_dist = (diff ** 2).sum(dim=2)              # (N, K)
        
        # Per-ball soft membership: (N, K)
        r2          = (r ** 2).unsqueeze(0)           # (1, K)
        ball_member = torch.exp(-sq_dist / (r2 + 1e-8))
        
        # Union probability: 1 - prod(1 - ball_member)
        # For numerical stability use log-space
        log_not_member = torch.log(1 - ball_member + 1e-8)  # (N, K)
        log_prod       = log_not_member.sum(dim=1)            # (N,)
        union_score    = 1 - torch.exp(log_prod)              # (N,)
        
        return union_score

    def forward(self, h):
        '''
        h: (N, d) penultimate representation
        Returns:
            p_pos : (N,) probability of minority class
            p_neg : (N,) probability of majority class
            s_pos : (N,) raw minority membership score
            s_neg : (N,) raw majority membership score
        '''
        r_pos, r_neg = self.get_radii()
        
        s_pos = self.union_of_balls(h, self.mu_pos, r_pos)  # (N,)
        s_neg = self.union_of_balls(h, self.mu_neg, r_neg)  # (N,)
        
        # Normalize to probabilities
        total = s_pos + s_neg + 1e-8
        p_pos = s_pos / total  # (N,)
        p_neg = s_neg / total  # (N,)
        
        return p_pos, p_neg, s_pos, s_neg


# ============================================================
# 2. FULL NETWORK WITH CCCD FINAL LAYER
# ============================================================

class NetCCCD(nn.Module):
    '''
    Standard hidden layers + CCCD final layer.
    The penultimate representation feeds into CCCDLayer.
    '''
    def __init__(self, p, h, n_layers, act_name, k_pos, k_neg):
        super(NetCCCD, self).__init__()
        
        act_dict = {
            'relu'    : nn.ReLU,
            'lrelu'   : lambda: nn.LeakyReLU(0.2),
            'tanh'    : nn.Tanh,
            'sigmoid' : nn.Sigmoid
        }
        act_fn = act_dict.get(act_name.lower(), nn.ReLU)
        
        # Build hidden layers
        layers = []
        in_f   = int(p)
        h_f    = int(h)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_f, h_f))
            layers.append(act_fn())
            in_f = h_f
        self.hidden = nn.Sequential(*layers)
        
        # CCCD final layer operates on penultimate dim = h_f
        self.cccd = CCCDLayer(d=h_f, k_pos=int(k_pos), k_neg=int(k_neg))

    def forward(self, x):
        h = self.hidden(x)                          # (N, h_f)
        p_pos, p_neg, s_pos, s_neg = self.cccd(h)
        return p_pos, p_neg, s_pos, s_neg


# ============================================================
# 3. CCCD LOSS
# ============================================================

class CCCDLoss(nn.Module):
    '''
    Three-term loss:
    L = L_cls + lambda1 * L_cover + lambda2 * L_repel + lambda3 * L_penalty
    
    L_cls    : NLL on soft membership probabilities
    L_cover  : maximize correct class coverage
    L_repel  : minimize wrong class coverage  
    L_penalty: L2 penalty on radii (controls layer effect)
    '''
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.1):
        super(CCCDLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, p_pos, p_neg, s_pos, s_neg, targets, cccd_layer):
        '''
        p_pos, p_neg : (N,) normalized class probabilities
        s_pos, s_neg : (N,) raw membership scores
        targets      : (N,) binary {0=majority, 1=minority}
        cccd_layer   : CCCDLayer instance (for radii access)
        '''
        targets = targets.squeeze(1)   # (N,)
        
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)

        # --- L_cls: NLL ---
        # p_pos is probability of minority, p_neg of majority
        p_true = torch.where(pos_mask, p_pos, p_neg)
        L_cls  = -torch.log(p_true + 1e-8).mean()

        # --- L_cover: maximize correct class coverage ---
        L_cover = 0.0
        if pos_mask.sum() > 0:
            L_cover = L_cover - s_pos[pos_mask].mean()
        if neg_mask.sum() > 0:
            L_cover = L_cover - s_neg[neg_mask].mean()

        # --- L_repel: minimize wrong class coverage ---
        L_repel = 0.0
        if pos_mask.sum() > 0:
            L_repel = L_repel + s_neg[pos_mask].mean()
        if neg_mask.sum() > 0:
            L_repel = L_repel + s_pos[neg_mask].mean()

        # --- L_penalty: L2 on radii ---
        r_pos, r_neg = cccd_layer.get_radii()
        L_penalty    = (r_pos ** 2).sum() + (r_neg ** 2).sum()

        # --- Total loss ---
        loss = (L_cls
                + self.lambda1 * L_cover
                + self.lambda2 * L_repel
                + self.lambda3 * L_penalty)

        return loss


# ============================================================
# 4. INITIALIZATION
# ============================================================

def init_dominant_points(x_train, y_train, k_pos, k_neg):
    '''
    Initialize dominant points using k-means on each class.
    Falls back to random sample selection if k > n_samples.
    '''
    from sklearn.cluster import KMeans
    
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32).flatten()

    x_pos = x_train[y_train == 1]
    x_neg = x_train[y_train == 0]
    
    def get_centers(x, k):
        k = min(k, len(x))
        if k == 1:
            return x[np.random.choice(len(x), 1)]
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(x)
        return km.cluster_centers_

    centers_pos = get_centers(x_pos, k_pos)
    centers_neg = get_centers(x_neg, k_neg)

    return centers_pos, centers_neg


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_cccd(x_train, y_train, p, h, n_layers, act_name,
               epochs, batch_size, lr,
               k_pos, k_neg,
               lambda1, lambda2, lambda3,
               device_name, patience=10):

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32).flatten()

    device  = torch.device(device_name)

    # --- Build model ---
    model = NetCCCD(p, h, n_layers, act_name,
                    k_pos=k_pos, k_neg=k_neg).to(device)

    # --- Initialize dominant points via forward pass ---
    x_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        h_all = model.hidden(x_t).cpu().numpy()

    from sklearn.cluster import KMeans
    y_int = y_train.astype(int)
    h_pos = h_all[y_int == 1]
    h_neg = h_all[y_int == 0]

    def get_centers(h, k):
        k = max(1, min(int(k), len(h)))
        if k == 1:
            return h[np.random.choice(len(h), 1)]
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(h)
        return km.cluster_centers_

    centers_pos = get_centers(h_pos, int(k_pos))
    centers_neg = get_centers(h_neg, int(k_neg))

    with torch.no_grad():
        model.cccd.mu_pos.copy_(
            torch.tensor(centers_pos, dtype=torch.float32).to(device))
        model.cccd.mu_neg.copy_(
            torch.tensor(centers_neg, dtype=torch.float32).to(device))

    # --- Stratified validation split ---
    pos_idx   = np.where(y_int == 1)[0]
    neg_idx   = np.where(y_int == 0)[0]
    n_val_pos = max(1, int(len(pos_idx) * 0.1))
    n_val_neg = max(1, int(len(neg_idx) * 0.1))
    val_idx   = np.concatenate([
        np.random.choice(pos_idx, n_val_pos, replace=False),
        np.random.choice(neg_idx, n_val_neg, replace=False)
    ])
    train_idx = np.setdiff1d(np.arange(len(x_train)), val_idx)

    x_tr = torch.tensor(x_train[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32).reshape(-1, 1)
    x_vl = torch.tensor(x_train[val_idx],   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_train[val_idx],   dtype=torch.float32).reshape(-1, 1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CCCDLoss(lambda1=lambda1,
                         lambda2=lambda2,
                         lambda3=lambda3)

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
            xb  = x_tr[ids].to(device)
            yb  = y_tr[ids].to(device)
            optimizer.zero_grad()
            p_pos, p_neg, s_pos, s_neg = model(xb)
            loss = criterion(p_pos, p_neg, s_pos, s_neg, yb, model.cccd)
            loss.backward()
            optimizer.step()

        # --- Validation loss ---
        model.eval()
        with torch.no_grad():
            p_pos_v, p_neg_v, s_pos_v, s_neg_v = model(x_vl)
            val_loss = criterion(p_pos_v, p_neg_v, s_pos_v, s_neg_v,
                                 y_vl, model.cccd).item()
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


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict_cccd(model, x_test, device_name):
    device = torch.device(device_name)
    model.eval()
    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        p_pos, p_neg, s_pos, s_neg = model(x_t)
        probs = p_pos.cpu().numpy()
    return probs.flatten()
")


classifier_CCCDNNET <- R6Class(
  "classifier_CCCDNNET",
  public = list(
    model       = NULL,
    device      = "cpu",
    class_names = NULL,
    class_pos   = NULL,
    
    initialize = function() {
      torch <- import("torch")
      self$device <- if (torch$cuda$is_available()) "cuda" else "cpu"
    },
    
    fit = function(x, y,
                   hidden_dim = 32L,
                   n_hidden   = 2L,
                   activation = "relu",
                   epochs     = 50L,
                   batch_size = 64L,
                   lr         = 1e-3,
                   k_pos      = NULL,
                   k_neg      = NULL,
                   lambda1    = 1.0,
                   lambda2    = 1.0,
                   lambda3    = 0.1,
                   patience   = 10,
                   alpha      = NULL,
                   k_min      = 1L) {
      
      x_mat <- as.matrix(x)
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)
      
      # Default K: floor(sqrt(n_class)); with alpha: K_c = min(n_c, max(k_min, floor(alpha * sqrt(n_c))))
      n_pos <- sum(y_bin == 1)
      n_neg <- sum(y_bin == 0)
      k_rule <- function(n) if (is.null(alpha)) max(1L, floor(sqrt(n))) else min(n, max(k_min, floor(alpha * sqrt(n))))
      if (is.null(k_pos)) k_pos <- k_rule(n_pos)
      if (is.null(k_neg)) k_neg <- k_rule(n_neg)
      
      self$model <- py$train_cccd(
        x_train    = x_mat,
        y_train    = y_bin,
        p          = ncol(x_mat),
        h          = hidden_dim,
        n_layers   = n_hidden,
        act_name   = activation,
        epochs     = epochs,
        batch_size = batch_size,
        lr         = lr,
        k_pos      = as.integer(k_pos),
        k_neg      = as.integer(k_neg),
        lambda1    = lambda1,
        lambda2    = lambda2,
        lambda3    = lambda3,
        device_name = self$device,
        patience   = patience
      )
      invisible(self)
    },
    
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
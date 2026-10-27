resamplerCls_SSNNetO <- function(x, y, lr = 0.001, batch_size = 32L, epochs = 100L, 
                                      hidden_dim = 64L, kl_weight = 0.0001, 
                                      init_mu = 0.1, init_logvar = -5.0, 
                                      ovRate = 1, ...) {
  library(FNN)
  torch <- import("torch")
  
  # --- 1. R SIDE: DATA PREP ---
  x_mat <- as.matrix(x)
  cls <- levels(y)
  x_pos <- x_mat[y == cls[2], , drop = FALSE]
  x_neg <- x_mat[y == cls[1], , drop = FALSE]
  p <- ncol(x_pos)
  
  nn <- get.knnx(data = x_neg, query = x_pos, k = 1)
  n_6 <- x_neg[nn$nn.index, , drop = FALSE]
  
  # Calculate Radius for Heteroscedastic Regularization
  radius <- sqrt(rowSums((x_pos - n_6)^2))
  
  # 4-Pole Mirroring
  m_12 <- 2 * x_pos - n_6
  v_axis <- x_pos - n_6
  v_ortho <- v_axis
  if(p >= 2) {
    v_ortho[, 1] <- -v_axis[, 2]
    v_ortho[, 2] <-  v_axis[, 1]
  }
  m_3 <- x_pos + v_ortho
  m_9 <- x_pos - v_ortho
  
  long_features <- rbind(n_6, m_12, m_3, m_9)
  long_targets <- rbind(x_pos, x_pos, x_pos, x_pos)
  long_radius <- rep(radius, 4)
  
  # --- 2. PYTHON SIDE ---
  py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossBNN(nn.Module):
    def __init__(self, p, hidden, i_mu, i_lv):
        super().__init__()
        self.encoder = nn.Linear(p, hidden)
        self.hidden = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, p)
        self.lv_head = nn.Linear(hidden, p)
        
        for layer in [self.encoder, self.hidden, self.mu_head]:
            nn.init.normal_(layer.weight, mean=0.0, std=i_mu)
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.constant_(self.lv_head.weight, 0.0)
        nn.init.constant_(self.lv_head.bias, float(i_lv))

    def forward(self, x):
        h = F.relu(self.encoder(x))
        h = F.relu(self.hidden(h))
        return self.mu_head(h), self.lv_head(h)

def train_cross(feat, targ, rad, n_gen, hidden, lr, epochs, bs, kl_w, i_mu, i_lv, device_str):
    dev = torch.device(device_str)
    f_t = torch.tensor(feat, dtype=torch.float32).to(dev)
    t_t = torch.tensor(targ, dtype=torch.float32).to(dev)
    r_t = torch.tensor(rad, dtype=torch.float32).to(dev)
    
    model = CrossBNN(t_t.shape[1], hidden, i_mu, i_lv).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    final_mse = 0.0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(f_t.size(0))
        epoch_mse = 0
        batches = 0
        for i in range(0, f_t.size(0), bs):
            sel = perm[i:i+bs]
            if sel.size(0) < 2: continue
            optimizer.zero_grad()
            
            mu, lv = model(f_t[sel])
            std = torch.exp(0.5 * lv)
            pred = mu + torch.randn_like(std) * std
            
            mse_loss = F.mse_loss(pred, t_t[sel])
            
            rad_col = r_t[sel].unsqueeze(1)
            dist = torch.sqrt(torch.sum((pred - t_t[sel].unsqueeze(1))**2, dim=1))
            violation = F.relu(dist - rad_col)
            
            # rad_col = r_t[sel].unsqueeze(1)
            # tau = torch.mean(rad_col) + 1e-6 
            # 
            # pressure = (rad_col / tau).pow(10)
            kl_penalty = torch.mean(violation * lv.pow(2))
            
            loss = mse_loss + kl_w * kl_penalty
            loss.backward()
            optimizer.step()
            
            epoch_mse += mse_loss.item()
            batches += 1
        
        final_mse = epoch_mse / batches

    model.eval()
    with torch.no_grad():
        p_orig = t_t[:int(t_t.size(0)/4)]
        seeds = torch.randint(0, p_orig.size(0), (int(n_gen),))
        mu, lv = model(p_orig[seeds])
        gen = (mu + torch.randn_like(mu) * torch.exp(0.5 * lv)).cpu().numpy()
        return {'gen': gen, 'mse': final_mse}
")
  
  # --- 3. EXECUTION ---
  n_need <- round((max(table(y)) - table(y)) * ovRate)
  dev_str <- if(torch$cuda$is_available()) "cuda" else "cpu"
  
  py_res <- py$train_cross(long_features, long_targets, long_radius, 
                           as.integer(n_need[2]), as.integer(hidden_dim), 
                           lr, as.integer(epochs), as.integer(batch_size), 
                           kl_weight, init_mu, init_logvar, dev_str)
  
  x_syn <- py_res$gen
  y_syn <- factor(rep(cls[2], nrow(x_syn)), levels = cls)
  
  return(list(
    x_new = rbind(x_mat, x_syn),
    y_new = factor(c(as.character(y), rep(cls[2], nrow(x_syn))), levels = cls),
    x_syn = x_syn,
    y_syn = y_syn,
    mse = py_res$mse
  ))
}
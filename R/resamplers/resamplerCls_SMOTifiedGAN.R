library(reticulate)

resamplerCls_SMOTifiedGAN <- function(x, y, lr = 0.00001, batch_size = 128L, epochs = 1000L,
                                      hidden_dim = 128L, noise_dim = 16L, k = 5L,
                                      ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        # Architecture per paper: ReLU + BatchNorm
        self.gen = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x_smote):
        # Your Style: Residual logic
        # G takes SMOTE samples as input and refines them
        delta = self.gen(x_smote)
        return delta + x_smote

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        # Paper Style: LeakyReLU(0.2)
        self.dis = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)

def smote_repertoire(x_min, n_needed, k):
    n_samples, n_feats = x_min.shape
    k = min(k, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=k+1).fit(x_min)
    _, indices = nn.kneighbors(x_min)
    
    synthetic = []
    for _ in range(n_needed):
        idx = np.random.randint(0, n_samples)
        neighbor_idx = np.random.choice(indices[idx][1:])
        diff = x_min[neighbor_idx] - x_min[idx]
        synthetic.append(x_min[idx] + np.random.rand() * diff)
    return np.array(synthetic)

def train_smotified(x_min_np, n_needed, epochs, batch_size, lr, hidden_dim, k, device_str):
    device = torch.device(device_str)
    x_real = torch.tensor(x_min_np, dtype=torch.float32).to(device)
    p = x_real.shape[1]
    
    # Phase 1: Generate SMOTE Repertoire
    u_np = smote_repertoire(x_min_np, n_needed, k)
    u_smote = torch.tensor(u_np, dtype=torch.float32).to(device)
    
    # Phase 2: GAN Refinement
    G = Generator(p, hidden_dim, p).to(device)
    D = Discriminator(p, hidden_dim).to(device)
    
    optim_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # Sample batches
        idx_r = torch.randint(0, x_real.size(0), (batch_size,))
        idx_u = torch.randint(0, u_smote.size(0), (batch_size,))
        
        real_batch = x_real[idx_r]
        smote_batch = u_smote[idx_u]

        # Train Discriminator
        optim_D.zero_grad()
        fake_batch = G(smote_batch).detach()
        loss_d = criterion(D(real_batch).view(-1), torch.ones(batch_size, device=device)) + criterion(D(fake_batch).view(-1), torch.zeros(batch_size, device=device))
        loss_d.backward()
        optim_D.step()

        # Train Generator
        optim_G.zero_grad()
        loss_g = criterion(D(G(smote_batch)).view(-1), torch.ones(batch_size, device=device))
        loss_g.backward()
        optim_G.step()

    G.eval()
    with torch.no_grad():
        # Final refinement of the SMOTE repertoire
        return G(u_smote).cpu().numpy()
")
  
  x_mat <- as.matrix(x); class_names <- levels(y); n_classes <- table(y)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  device_str <- if(torch$cuda$is_available()) "cuda" else "cpu"
  
  x_syn_list <- list(); y_syn_list <- list()
  
  for (cls in class_names) {
    needed <- as.integer(n_needed[cls])
    if (needed <= 0) next
    
    x_syn_np <- py$train_smotified(x_mat[y == cls, , drop=F], needed, as.integer(epochs), 
                                   as.integer(batch_size), lr, as.integer(hidden_dim), 
                                   as.integer(k), device_str)
    
    x_syn_list[[cls]] <- x_syn_np
    y_syn_list[[cls]] <- factor(rep(cls, nrow(x_syn_np)), levels = levels(y))
  }
  
  return(list(
    x_new = rbind(x_mat, do.call(rbind, x_syn_list)),
    y_new = c(y, do.call(c, y_syn_list)),
    x_syn = do.call(rbind, x_syn_list),
    y_syn = do.call(c, y_syn_list)
  ))
}
library(reticulate)

resamplerCls_ADA_INCVAE <- function(x, y, lr = 1e-4, batch_size = 64L, epochs = 500L, 
                                        hidden_dim = 32L, latent_dim = 4L, ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SimpleVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SimpleVAE, self).__init__()
        # Back to your original working architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        delta = self.decoder(z)
        return delta + x, mu, logvar

def train_ada_vae_simple(x_min, x_maj, n_needed, latent_dim, hidden_dim, lr, epochs, batch_size, device_str):
    device = torch.device(device_str)
    x_tensor = torch.tensor(x_min, dtype=torch.float32).to(device)
    p = x_tensor.shape[1]
    
    # Adaptive Weighting (The only part kept from the paper)
    nn_maj = NearestNeighbors(n_neighbors=1).fit(x_maj)
    dist_maj, _ = nn_maj.kneighbors(x_min)
    dist_maj = torch.tensor(dist_maj, dtype=torch.float32).to(device)
    # Weight borderline points higher
    weights = 1.0 / (dist_maj + 1e-5)
    weights = (weights / weights.mean()).view(-1)
    
    model = SimpleVAE(p, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(x_tensor.size(0))
        for i in range(0, x_tensor.size(0), batch_size):
            batch_idx = idx[i : i + batch_size]
            if len(batch_idx) < 2: continue
            
            batch = x_tensor[batch_idx]
            batch_w = weights[batch_idx]
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            
            # Weighted MSE + KLD
            mse_loss = (batch_w * F.mse_loss(recon, batch, reduction='none').sum(dim=1)).mean()
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            (mse_loss + kld_loss).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        idx_seeds = torch.randint(0, x_tensor.size(0), (int(n_needed),))
        z = torch.randn(int(n_needed), latent_dim, device=device)
        return (model.decoder(z) + x_tensor[idx_seeds]).cpu().numpy()
")
  
  x_mat <- as.matrix(x)
  class_names <- levels(y)
  x_min <- x_mat[y == class_names[2], , drop=F]
  x_maj <- x_mat[y == class_names[1], , drop=F]
  
  n_classes <- table(y)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  device_str <- if(torch$cuda$is_available()) "cuda" else "cpu"
  
  x_gen <- py$train_ada_vae_simple(x_min, x_maj, as.integer(n_needed[2]), as.integer(latent_dim), 
                                   as.integer(hidden_dim), lr, as.integer(epochs), 
                                   as.integer(batch_size), device_str)
  
  return(list(x_new = rbind(x_mat, x_gen), y_new = c(y, factor(rep(class_names[2], nrow(x_gen)), levels = levels(y))),
              x_syn = x_gen, y_syn = factor(rep(class_names[2], nrow(x_gen)), levels = levels(y))))
}
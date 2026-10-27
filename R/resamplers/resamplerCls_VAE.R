library(reticulate)

resamplerCls_VAE <- function(x, y, lr = 0.001, batch_size = 64L, epochs = 500L, 
                             hidden_dim = 32L, latent_dim = 4L, ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder: Maps input to mean and log-variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: Maps latent vector back to delta
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
        return delta + x, mu, logvar # Your Style: Residual return

def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence: forces latent space to be normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_vae(x_main, n_needed, latent_dim, hidden_dim, lr, epochs, batch_size, device_str):
    device = torch.device(device_str)
    x_tensor = torch.tensor(x_main, dtype=torch.float32).to(device)
    p = x_tensor.shape[1]
    
    model = VAE(p, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(x_tensor.size(0))
        for i in range(0, x_tensor.size(0), batch_size):
            batch = x_tensor[idx[i : i + batch_size]]
            if batch.size(0) < 2: continue
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        # Generation: Sample z from N(0,1) and pass through Decoder
        # We still use real samples as anchors for the Residual logic
        idx_seeds = torch.randint(0, x_tensor.size(0), (int(n_needed),))
        x_seeds = x_tensor[idx_seeds]
        z = torch.randn(int(n_needed), latent_dim, device=device)
        deltas = model.decoder(z)
        return (deltas + x_seeds).cpu().numpy()
")
  
  x_mat <- as.matrix(x)
  class_names <- levels(y)
  n_classes <- table(y)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  device_str <- if(torch$cuda$is_available()) "cuda" else "cpu"
  
  x_syn_list <- list()
  for (cl in class_names) {
    needed <- as.integer(n_needed[cl])
    if (needed <= 0) next
    x_gen <- py$train_vae(x_mat[y == cl, , drop=F], needed, as.integer(latent_dim), 
                          as.integer(hidden_dim), lr, as.integer(epochs), 
                          as.integer(batch_size), device_str)
    x_syn_list[[cl]] <- x_gen
  }
  
  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(rep(names(x_syn_list), sapply(x_syn_list, nrow)), levels = levels(y))
  return(list(x_new = rbind(x_mat, x_syn), y_new = c(y, y_syn), x_syn = x_syn, y_syn = y_syn))
}
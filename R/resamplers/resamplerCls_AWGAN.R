library(reticulate)

resamplerCls_AWGAN <- function(x, y, lr = 0.0001, batch_size = 64L, epochs = 100L,
                               hidden_dim = 32L, noise_dim = 16L, lambda_1 = 10,
                               ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.relu_f = nn.ReLU(True)
        # Strictly using your hidden_dim and noise_dim
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, x_real):
        # Author's Residual Logic: Noise-driven delta + Real Anchor
        delta = self.gen(z)
        return self.relu_f(delta + x_real)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.dis(x)

def calculate_gradient_penalty(real_data, fake_data, D, center=1, p=2):
    eta = torch.rand(real_data.size(0), 1, device=real_data.device).expand_as(real_data)
    interpolated = (eta * real_data + ((1 - eta) * fake_data)).requires_grad_(True)
    prob_interpolated = D(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True, retain_graph=True)[0]
    return ((gradients.norm(2, dim=1) - center) ** p).mean()

def train_and_generate(x_main_np, n_needed, epochs, batch_size, lr, 
                       noise_dim, hidden_dim, lambda_1, device_str):
    device = torch.device(device_str)
    x_real = torch.tensor(x_main_np, dtype=torch.float32).to(device)
    p = x_real.shape[1]
    
    G = Generator(noise_dim, hidden_dim, p).to(device)
    D = Discriminator(p, hidden_dim).to(device)
    
    optim_G = torch.optim.AdamW(G.parameters(), lr=lr)
    optim_D = torch.optim.AdamW(D.parameters(), lr=lr)
    
    for epoch in range(epochs):
        idx = torch.randperm(x_real.size(0))
        for i in range(0, x_real.size(0), batch_size):
            batch_idx = idx[i : i + batch_size]
            real_batch = x_real[batch_idx]
            b_size = real_batch.size(0)
            if b_size < 2: continue

            # Train Discriminator
            optim_D.zero_grad()
            z = torch.randn(b_size, noise_dim, device=device)
            fake_batch = G(z, real_batch).detach()
            
            d_loss = -torch.mean(D(real_batch)) + torch.mean(D(fake_batch)) + calculate_gradient_penalty(real_batch, fake_batch, D) / lambda_1
            d_loss.backward()
            optim_D.step()

            # Train Generator
            optim_G.zero_grad()
            z = torch.randn(b_size, noise_dim, device=device)
            g_loss = -torch.mean(D(G(z, real_batch)))
            g_loss.backward()
            optim_G.step()

    G.eval()
    with torch.no_grad():
        # Sample real indices to provide the residual anchor for generation
        s_idx = torch.randint(0, x_real.size(0), (n_needed,))
        z = torch.randn(n_needed, noise_dim, device=device)
        return G(z, x_real[s_idx]).cpu().numpy()
")
  
  # --- R-Side Controller ---
  x_mat <- as.matrix(x)
  class_names <- levels(y)
  n_classes <- table(y)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  device_str <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  
  x_syn_list <- list()
  y_syn_list <- list()
  
  for (cls in class_names) {
    needed <- as.integer(n_needed[cls])
    if (needed <= 0) next
    
    x_syn_np <- py$train_and_generate(
      x_main_np = x_mat[y == cls, , drop = FALSE],
      n_needed = needed,
      epochs = as.integer(epochs),
      batch_size = as.integer(batch_size),
      lr = lr,
      noise_dim = as.integer(noise_dim),
      hidden_dim = as.integer(hidden_dim),
      lambda_1 = lambda_1,
      device_str = device_str
    )
    
    x_syn_list[[cls]] <- x_syn_np
    y_syn_list[[cls]] <- factor(rep(cls, nrow(x_syn_np)), levels = levels(y))
  }
  
  # Return everything including the requested x_syn and y_syn
  return(list(
    x_new = rbind(x_mat, do.call(rbind, x_syn_list)),
    y_new = c(y, do.call(c, y_syn_list)),
    x_syn = do.call(rbind, x_syn_list),
    y_syn = do.call(c, y_syn_list)
  ))
}
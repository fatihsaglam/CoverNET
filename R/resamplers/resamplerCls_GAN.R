library(reticulate)

resamplerCls_GAN <- function(x, y, lr = 0.001, batch_size = 64L, epochs = 100L, 
                             hidden_dim = 16L, noise_dim = 16L, ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, z, x_real):
        # RESIDUAL LOGIC: Learning the delta from a real point
        delta = self.net(z)
        return delta + x_real

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_gan_residual(x_main, n_needed, noise_dim, hidden_dim, lr, epochs, batch_size, device_str):
    device = torch.device(device_str)
    x_tensor = torch.tensor(x_main, dtype=torch.float32).to(device)
    p = x_tensor.shape[1] # Dynamically get number of features
    
    netG = Generator(noise_dim, hidden_dim, p).to(device)
    netD = Discriminator(p, hidden_dim).to(device)
    
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        idx = torch.randperm(x_tensor.size(0))
        for i in range(0, x_tensor.size(0), batch_size):
            # Batch preparation
            real_batch = x_tensor[idx[i : i + batch_size]]
            b_size = real_batch.size(0)
            if b_size < 2: continue # BatchNorm needs batch > 1
            
            real_labels = torch.ones(b_size, 1, device=device)
            fake_labels = torch.zeros(b_size, 1, device=device)

            # 1. Train Discriminator
            optimD.zero_grad()
            z = torch.randn(b_size, noise_dim, device=device)
            fake_batch = netG(z, real_batch) # Pass real_batch anchor
            
            loss_D = (criterion(netD(real_batch), real_labels) + 
                      criterion(netD(fake_batch.detach()), fake_labels)) / 2
            loss_D.backward()
            optimD.step()

            # 2. Train Generator
            optimG.zero_grad()
            loss_G = criterion(netD(fake_batch), real_labels)
            loss_G.backward()
            optimG.step()

    # Final Generation using Residual logic
    netG.eval()
    with torch.no_grad():
        # Select random seeds from original data to add delta to
        idx_seeds = torch.randint(0, x_tensor.size(0), (int(n_needed),))
        x_seeds = x_tensor[idx_seeds]
        z_final = torch.randn(int(n_needed), noise_dim, device=device)
        return netG(z_final, x_seeds).cpu().numpy()
")
  
  x_mat <- as.matrix(x)
  class_names <- levels(y)
  n_classes <- table(y)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  device_str <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  
  x_syn_list <- list(); y_syn_list <- list()
  
  for (cl in class_names) {
    needed <- as.integer(n_needed[cl])
    if (needed <= 0) next
    
    x_gen <- py$train_gan_residual(
      x_main = x_mat[y == cl, , drop = FALSE],
      n_needed = needed,
      noise_dim = as.integer(noise_dim),
      hidden_dim = as.integer(hidden_dim),
      lr = lr,
      epochs = as.integer(epochs),
      batch_size = as.integer(batch_size),
      device_str = device_str
    )
    
    x_syn_list[[cl]] <- x_gen
    y_syn_list[[cl]] <- factor(rep(cl, needed), levels = levels(y))
  }
  
  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- do.call(c, y_syn_list)
  
  return(list(x_new = rbind(x_mat, x_syn), y_new = c(y, y_syn), x_syn = x_syn, y_syn = y_syn))
}
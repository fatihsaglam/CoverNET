library(reticulate)

resamplerCls_WGANGP <- function(x, y, lr = 1e-4, batch_size = 64L, epochs = 1000L, 
                                hidden_dim = 64L, noise_dim = 128L, lambda_gp = 10,
                                critic_iters = 5L, ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    def forward(self, z, x_real):
        # Your 'Secret Sauce': Residual connection for tabular stability
        return self.net(z) + x_real

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).view(-1)

def calc_gradient_penalty(netD, real_data, fake_data, lambda_gp, device):
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand(real_data.size())
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def train_wgan_gp(x_main, n_needed, noise_dim, hidden_dim, lr, epochs, batch_size, lambda_gp, critic_iters, device_str):
    device = torch.device(device_str)
    x_tensor = torch.tensor(x_main, dtype=torch.float32).to(device)
    p = x_tensor.shape[1]
    
    netG = Generator(noise_dim, hidden_dim, p).to(device)
    netD = Discriminator(p, hidden_dim).to(device)
    
    # WGAN-GP uses Adam with specific betas (0, 0.9) or (0.5, 0.9)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for _ in range(critic_iters):
            idx = torch.randperm(x_tensor.size(0))[:batch_size]
            real_batch = x_tensor[idx]
            b_size = real_batch.size(0)
            if b_size < 2: continue

            netD.zero_grad()
            z = torch.randn(b_size, noise_dim, device=device)
            fake_batch = netG(z, real_batch)
            
            # WGAN Loss: D(fake) - D(real)
            loss_D = netD(fake_batch).mean() - netD(real_batch).mean()
            gp = calc_gradient_penalty(netD, real_batch.data, fake_batch.data, lambda_gp, device)
            (loss_D + gp).backward()
            optimizerD.step()

        # Update Generator
        netG.zero_grad()
        z = torch.randn(batch_size, noise_dim, device=device)
        # Sample random anchors for the Generator update
        idx_anchor = torch.randperm(x_tensor.size(0))[:batch_size]
        loss_G = -netD(netG(z, x_tensor[idx_anchor])).mean()
        loss_G.backward()
        optimizerG.step()

    netG.eval()
    with torch.no_grad():
        idx_seeds = torch.randint(0, x_tensor.size(0), (int(n_needed),))
        z_final = torch.randn(int(n_needed), noise_dim, device=device)
        return netG(z_final, x_tensor[idx_seeds]).cpu().numpy()
")
  
  # ... (Same R logic as your GAN/DAGAN resamplers) ...
  x_mat <- as.matrix(x); class_names <- levels(y); n_classes <- table(y)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  device_str <- if(torch$cuda$is_available()) "cuda" else "cpu"
  x_syn_list <- list()
  
  for (cl in class_names) {
    needed <- as.integer(n_needed[cl])
    if (needed <= 0) next
    x_gen <- py$train_wgan_gp(x_mat[y == cl, , drop=F], needed, as.integer(noise_dim), 
                              as.integer(hidden_dim), lr, as.integer(epochs), 
                              min(nrow(x_mat[y == cl, , drop=F]), as.integer(batch_size)), lambda_gp, as.integer(critic_iters), device_str)
    x_syn_list[[cl]] <- x_gen
  }
  
  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(rep(names(x_syn_list), sapply(x_syn_list, nrow)), levels = levels(y))
  return(list(x_new = rbind(x_mat, x_syn), y_new = c(y, y_syn), x_syn = x_syn, y_syn = y_syn))
}
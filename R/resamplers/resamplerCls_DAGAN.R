library(reticulate)

resamplerCls_DAGAN <- function(x, y, lr = 0.0002, batch_size = 64L, epochs = 200L,
                               hidden_dim = 64L, noise_dim = 100L, lambda_mmd = 0.1,
                               ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        # Paper uses standard ReLU and BatchNorm for internal layers
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, x_real):
        # Residual connection as per your successful tuning
        # No final ReLU to support negative data
        delta = self.gen(z)
        return delta + x_real

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        # Paper uses LeakyReLU (0.2) for the Discriminator
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.features(x)
        # Paper uses Sigmoid for the final classification probability
        validity = torch.sigmoid(self.classifier(feat))
        return validity, feat

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)

def mmd_loss(source, target):
    batch_size = int(source.size(0))
    kernels = gaussian_kernel(source, target)
    loss = torch.mean(kernels[:batch_size, :batch_size]) + torch.mean(kernels[batch_size:, batch_size:]) - 2 * torch.mean(kernels[:batch_size, batch_size:])
    return loss

def train_dagan(x_source, x_target, n_needed, epochs, batch_size, lr, 
                noise_dim, hidden_dim, lambda_mmd, device_str):
    device = torch.device(device_str)
    src = torch.tensor(x_source, dtype=torch.float32).to(device)
    tgt = torch.tensor(x_target, dtype=torch.float32).to(device)
    p = src.shape[1]
    
    G = Generator(noise_dim, hidden_dim, p).to(device)
    D = Discriminator(p, hidden_dim).to(device)
    
    # Paper uses standard Adam
    optim_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        # Shuffle and sample
        idx_s = torch.randperm(src.size(0))[:batch_size]
        idx_t = torch.randperm(tgt.size(0))[:batch_size]
        if idx_s.size(0) < 2 or idx_t.size(0) < 2: continue
        
        real_s = src[idx_s]
        real_t = tgt[idx_t]
        
        # --- Discriminator ---
        optim_D.zero_grad()
        z = torch.randn(real_t.size(0), noise_dim, device=device)
        fake_t = G(z, real_t).detach()
        
        d_real, _ = D(real_t)
        d_fake, _ = D(fake_t)
        
        loss_D = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        loss_D.backward()
        optim_D.step()
        
        # --- Generator ---
        optim_G.zero_grad()
        z = torch.randn(real_t.size(0), noise_dim, device=device)
        gen_t = G(z, real_t)
        g_valid, feat_t = D(gen_t)
        _, feat_s = D(real_s)
        
        # Adversarial + Deep Domain Confusion (MMD)
        loss_adv = F.binary_cross_entropy(g_valid, torch.ones_like(g_valid))
        loss_mmd = mmd_loss(feat_s, feat_t)
        
        (loss_adv + lambda_mmd * loss_mmd).backward()
        optim_G.step()

    G.eval()
    with torch.no_grad():
        s_idx = torch.randint(0, tgt.size(0), (n_needed,))
        z = torch.randn(n_needed, noise_dim, device=device)
        return G(z, tgt[s_idx]).cpu().numpy()
")
  
  # --- R Controller ---
  x_mat <- as.matrix(x)
  class_names <- levels(y)
  n_classes <- table(y)
  major_cls <- names(which.max(n_classes))
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  device_str <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  
  x_syn_list <- list()
  y_syn_list <- list()
  
  for (cls in class_names) {
    if (cls == major_cls || n_needed[cls] <= 0) next
    
    x_syn_np <- py$train_dagan(
      x_source = x_mat[y == major_cls, , drop=FALSE],
      x_target = x_mat[y == cls, , drop=FALSE],
      n_needed = as.integer(n_needed[cls]),
      epochs = as.integer(epochs),
      batch_size = as.integer(batch_size),
      lr = lr,
      noise_dim = as.integer(noise_dim),
      hidden_dim = as.integer(hidden_dim),
      lambda_mmd = lambda_mmd,
      device_str = device_str
    )
    
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
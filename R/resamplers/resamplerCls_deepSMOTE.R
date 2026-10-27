library(reticulate)

resamplerCls_deepSMOTE <- function(x, y, k = 5L, lr = 0.01, batch_size = 32L, 
                                   epochs = 100L, hidden_dim = 16L, 
                                   latent_dim = 4L, ovRate = 1, ...) {
  
  torch <- import("torch")
  
  py_run_string("
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_logic(batches, encoder, decoder, optimizer, epochs, device):
    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # --- Standard Reconstruction (RL) ---
            EB = encoder(batch)
            DB = decoder(EB)
            RL = torch.mean((DB - batch)**2)
            
            # --- DeepSMOTE Penalty (PL) ---
            # Matches your R logic: permute the order of the batch
            idx = torch.randperm(batch.size(0), device=device)
            CD = batch[idx, :] # Shuffled version of the same class batch
            ES = encoder(CD)
            
            # Feature permutation (Matching your `permute_order` logic)
            feat_idx = torch.randperm(ES.size(1), device=device)
            PE = ES[:, feat_idx]
            
            DP = decoder(PE)
            PL = torch.mean((DP - CD)**2)
            
            TL = RL + PL
            TL.backward()
            optimizer.step()
")
  
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  
  # --- R Controller ---
  x <- as.matrix(x)
  p <- ncol(x)
  class_names <- levels(y)
  n_classes <- table(y)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  
  x_syn_list <- list()
  y_syn_list <- list()
  
  for (cls in class_names) {
    if (n_needed[cls] <= 0) next
    
    x_main_tensor <- torch$tensor(x[y == cls, , drop=FALSE], dtype=torch$float32)
    n <- x_main_tensor$size(0L)
    
    # Batching
    idx_perm <- torch$randperm(n)
    batches <- lapply(seq(1, n, by = batch_size), function(i) {
      end <- min(i + batch_size - 1, n)
      x_main_tensor[idx_perm[i:end], ]
    })
    
    enc <- py$Encoder(as.integer(p), as.integer(hidden_dim), as.integer(latent_dim))$to(device)
    dec <- py$Decoder(as.integer(latent_dim), as.integer(hidden_dim), as.integer(p))$to(device)
    optim <- torch$optim$Adam(c(iterate(enc$parameters()), iterate(dec$parameters())), lr = lr)
    
    # Train
    py$train_logic(batches, enc, dec, optim, as.integer(epochs), device)
    
    # Generate (Native R SMOTE step to ensure it matches your previous results)
    enc$eval(); dec$eval()
    with(torch$no_grad(), {
      embeddings <- enc(x_main_tensor$to(device))
      x_latent <- embeddings$cpu()$numpy()
    })
    
    nn_idx <- FNN::get.knnx(x_latent, x_latent, k = k + 1)$nn.index[,-1, drop=FALSE]
    x_syn_latent <- matrix(NA, nrow = n_needed[cls], ncol = ncol(x_latent))
    
    for(i in 1:n_needed[cls]) {
      idx <- sample(1:nrow(x_latent), 1)
      neighbor_idx <- sample(nn_idx[idx, ], 1)
      r <- runif(1)
      x_syn_latent[i,] <- x_latent[idx,] + r * (x_latent[neighbor_idx,] - x_latent[idx,])
    }
    
    x_syn_tensor <- torch$tensor(x_syn_latent, dtype=torch$float32)$to(device)
    x_syn_list[[cls]] <- dec(x_syn_tensor)$detach()$cpu()$numpy()
    y_syn_list[[cls]] <- factor(rep(cls, n_needed[cls]), levels = levels(y))
  }
  
  return(list(
    x_new = rbind(x, do.call(rbind, x_syn_list)),
    y_new = c(y, do.call(c, y_syn_list)),
    x_syn = do.call(rbind, x_syn_list),
    y_syn = do.call(c, y_syn_list)
  ))
}
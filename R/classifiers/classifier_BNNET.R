library(R6)
library(reticulate)

# Define the Bayesian Net in Python space
py_run_string("
import torch
import torch.nn as nn
import torchbnn as bnn
import numpy as np

class BayesianNet(nn.Module):
    def __init__(self, p, h, n_layers, prior_mu, prior_sigma):
        super(BayesianNet, self).__init__()
        layers = []
        in_f, h_f = int(p), int(h)
        for _ in range(int(n_layers)):
            layers.append(bnn.BayesLinear(prior_mu=float(prior_mu), prior_sigma=float(prior_sigma), 
                                          in_features=in_f, out_features=h_f))
            layers.append(nn.ReLU())
            in_f = h_f
        layers.append(bnn.BayesLinear(prior_mu=float(prior_mu), prior_sigma=float(prior_sigma), 
                                      in_features=in_f, out_features=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_bnnet_python(x_train, y_train, p, h, n_layers, prior_mu, prior_sigma, kl_weight, epochs, batch_size, lr, device_name):
    device = torch.device(device_name)
    model = BayesianNet(p, h, n_layers, prior_mu, prior_sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.BCEWithLogitsLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    model.train()
    n = len(x_train)
    for epoch in range(int(epochs)):
        indices = np.random.permutation(n)
        for i in range(0, n, int(batch_size)):
            ids = indices[i : i + int(batch_size)]
            xb, yb = x_t[ids].to(device), y_t[ids].to(device)
            optimizer.zero_grad()
            # BNN Loss: Data Likelihood + Complexity Cost
            loss = ce_loss(model(xb), yb) + (float(kl_weight) * kl_loss(model))
            loss.backward()
            optimizer.step()
            
    return model

def predict_bnnet_python(model, x_test, n_draws, device_name):
    device = torch.device(device_name)
    model.train() # Bayesian layers need to stay in train mode to sample weights
    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    
    outputs = []
    with torch.no_grad():
        for _ in range(int(n_draws)):
            logits = model(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            outputs.append(probs.flatten())
    
    return np.mean(outputs, axis=0)
")

classifier_BNNET <- R6Class(
  "classifier_BNNET",
  public = list(
    model = NULL,
    device = NULL,
    class_names = NULL,
    class_pos = NULL,
    
    initialize = function() {
      torch <- import("torch")
      # self$device <- if (torch$cuda$is_available()) "cuda" else "cpu"
      self$device <- "cpu"
    },
    
    fit = function(x, y, hidden_dim = 32L, n_hidden = 2L, 
                   prior_mu = 0.0, prior_sigma = 0.1, 
                   kl_weight = 0.1, epochs = 50L, 
                   batch_size = 64L, lr = 1e-2) {
      
      x_mat <- as.matrix(x)
      self$class_names <- levels(y)
      self$class_pos <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)
      
      # The entire training loop runs inside Python
      self$model <- py$train_bnnet_python(
        x_train     = x_mat, 
        y_train     = y_bin,
        p           = ncol(x_mat), 
        h           = hidden_dim, 
        n_layers    = n_hidden, 
        prior_mu    = prior_mu, 
        prior_sigma = prior_sigma,
        kl_weight   = kl_weight,
        epochs      = epochs, 
        batch_size  = batch_size, 
        lr          = lr, 
        device_name = self$device
      )
      
      invisible(self)
    },
    
    predict = function(x, n_draws = 50L) {
      avg_probs_pos <- py$predict_bnnet_python(
        model       = self$model, 
        x_test      = as.matrix(x), 
        n_draws     = n_draws, 
        device_name = self$device
      )
      
      # only binary for now
      prob <- matrix(NA, nrow = length(avg_probs_pos), ncol = 2)
      prob[,which(self$class_names == self$class_pos)] <- avg_probs_pos
      prob[,which(self$class_names != self$class_pos)] <- 1 - avg_probs_pos
      
      colnames(prob) <- self$class_names
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      
      return(list(pred = pred, prob = prob))
    }
  )
)
# library(FatihResearch)
library(reticulate)
np <- reticulate::import("numpy")
torch <- reticulate::import("torch")
nn <- import("torch.nn", convert = TRUE)

resamplerCls_deepMIMSMOTE <- function(x,
                                      y,
                                      k = 5L,
                                      lr = 0.001,
                                      batch_size = 32L,
                                      epochs = 100L,
                                      hidden_dim = 16L,
                                      latent_dim = 4L,
                                      ovRate = 1, ...) {
  Encoder <- PyClass(
    "Encoder",
    inherit = nn$Module,
    defs = list(
      `__init__` = function(self, input_dim, hidden_dim, latent_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(as.integer(input_dim), as.integer(hidden_dim))
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(as.integer(hidden_dim), as.integer(latent_dim))
        NULL
      },
      forward = function(self, x) {
        x <- self$fc1(x)
        x <- self$relu(x)
        x <- self$fc2(x)
        return(x)
      }
    )
  )
  
  Decoder <- PyClass(
    "Decoder",
    inherit = nn$Module,
    defs = list(
      `__init__` = function(self, latent_dim, hidden_dim, output_dim) {
        super()$`__init__`()
        self$fc1 <- nn$Linear(as.integer(latent_dim), as.integer(hidden_dim))
        self$relu <- nn$ReLU()
        self$fc2 <- nn$Linear(as.integer(hidden_dim), as.integer(output_dim))
        NULL
      },
      forward = function(self, x) {
        x <- self$fc1(x)
        x <- self$relu(x)
        x <- self$fc2(x)
        return(x)
      }
    )
  )
  
  permute_order <- function(x) {
    idx <- torch$randperm(x$size(1L))
    return(x[ , idx])
  }
  
  reconstruction_loss <- function(pred, target) {
    return(torch$mean((pred - target)$pow(2L)))
  }
  
  train_deepsmote <- function(data_loader, encoder, decoder, optimizer, epochs,
                              patience = epochs, min_delta = 1e-4) {
    
    best_loss <- Inf
    no_improve_count <- 0
    
    for (epoch in 1:epochs) {
      total_loss <- 0
      batch_count <- 0
      for (batch in data_loader) {
        if (batch$shape[0] < 1) {
          next
        }
        optimizer$zero_grad()
        EB <- encoder(batch)
        DB <- decoder(EB)
        
        CD <- batch
        ES <- encoder(CD)
        PE <- permute_order(ES)
        DP <- decoder(PE)
        
        PL <- reconstruction_loss(DP, CD)
        RL <- reconstruction_loss(DB, batch)
        TL <- RL + PL
        
        TL$backward()
        optimizer$step()
        
        total_loss <- total_loss + TL$item()
        batch_count <- batch_count + 1
      }
      
      avg_loss <- total_loss / batch_count

      # Early stopping condition
      if (avg_loss < best_loss - min_delta) {
        best_loss <- avg_loss
        no_improve_count <- 0
      } else {
        no_improve_count <- no_improve_count + 1
      }
      
    }
  }
  
  create_batches <- function(tensor_data, batch_size) {
    n <- tensor_data$size(0L)
    idx <- torch$randperm(n)
    batches <- list()
    for (i in seq(1, n, by = batch_size)) {
      end <- min(i + batch_size - 1, n)
      batches[[length(batches) + 1]] <- tensor_data[idx[i:end], ]
    }
    return(batches)
  }
  
  p <- ncol(x)
  x <- as.matrix(x)
  class_names <- levels(y)
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  n_needed <- round((max(n_classes) - n_classes) * ovRate)
  
  x_imp <- x
  x_ind <- is.na(x)*1
  m_mice <- mice::mice(data.frame(x_imp, y), printFlag = FALSE)
  x_imp <- complete(m_mice)[,1:p]
  x_mim <- cbind(x_imp, x_ind)
  
  x_syn_list <- sapply(1:k_class, function(m) list())
  
  for (i_class in 1:k_class) {
    if (n_needed[i_class] == 0) {
      x_syn_list[[i_class]] <- matrix(data = NA, nrow = 0, ncol = ncol(x_mim))
      next
    }
    
    x_main_mim <- x_mim[y == class_names[i_class],, drop = FALSE]
    x_other_mim <- x_mim[y != class_names[i_class],, drop = FALSE]
    
    x_main_mim_tensor <- torch$tensor(as.matrix(x_main_mim), dtype = torch$float32)
    batches <- create_batches(tensor_data = x_main_mim_tensor, batch_size = batch_size)
    
    input_dim <- ncol(x_main_mim)
    encoder <- Encoder(input_dim, hidden_dim, latent_dim)
    decoder <- Decoder(latent_dim, hidden_dim, input_dim)
    params <- c(
      iterate(encoder$parameters()),
      iterate(decoder$parameters())
    )
    optim <- torch$optim$RMSprop(params, lr = lr)
    
    train_deepsmote(data_loader = batches, encoder = encoder, decoder = decoder, optimizer = optim, epochs = epochs)
    
    x_main_mim_tensor_encoded <- encoder(x_main_mim_tensor)
    x_main_mim_encoded <- x_main_mim_tensor_encoded$detach()$numpy()
    
    NN_main2main <- FNN::get.knnx(data = x_main_mim_encoded, query = x_main_mim_encoded, k = k + 1)$nn.index[,-1, drop = FALSE]
    
    x_syn_main <- matrix(data = NA, nrow = 0, ncol = ncol(x_main_mim_encoded))
    
    while (nrow(x_syn_main) < n_needed[i_class]) {
      i_selected <- sample(1:n_classes[i_class], size = 1)
      x_center <- x_main_mim_encoded[i_selected,, drop = FALSE]
      x_other <- x_main_mim_encoded[sample(NN_main2main[i_selected,], size = 1),, drop = FALSE]
      r <- runif(1)
      x_syn_temp <- x_other - r*(x_other - x_center)
      x_syn_main <- rbind(x_syn_main, x_syn_temp)
    }
    
    x_syn_main_tensor <- torch$tensor(as.matrix(x_syn_main), dtype = torch$float32)
    x_syn_main_decoded_tensor <- decoder(x_syn_main_tensor)
    x_syn_main_decoded <- x_syn_main_decoded_tensor$detach()$numpy()
    
    m_NN_main2other <- FNN::get.knnx(data = x_other_mim, query = x_main_mim, k = 1)
    NN_main2other <- m_NN_main2other$nn.index
    d_main2other <- m_NN_main2other$nn.dist
    
    m_NN_syn2main <- FNN::get.knnx(data = x_main_mim, query = x_syn_main_decoded, k = 1)
    
    NN_syn2main <- m_NN_syn2main$nn.index
    d_syn2main <- m_NN_syn2main$nn.dist
    
    for (i in 1:n_needed[i_class]) {
      x_syn_selected <- x_syn_main_decoded[i,,drop = FALSE]
      i_nearest_main <- NN_syn2main[i]
      x_nearest_main <- x_main_mim[i_nearest_main,, drop = FALSE]
      
      radii <- d_main2other[i_nearest_main]
      d_syn2main_i <- d_syn2main[i]
      
      x_syn_selected + (x_nearest_main - x_syn_selected)
      r <- runif(n = 1, min = (d_syn2main_i - radii)/d_syn2main_i, 
            max = (d_syn2main_i + radii)/d_syn2main_i)
      
      x_syn_main_decoded[i,] <- unlist(c(x_syn_selected + r*(x_nearest_main - x_syn_selected)))
    }

    x_syn_list[[i_class]] <- x_syn_main_decoded
  }

  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(rep(class_names, n_needed), levels = class_names)
  
  x_new <- rbind(as.matrix(x_mim[,1:p, drop = FALSE]), x_syn[,1:p, drop = FALSE])
  y_new <- c(y, y_syn)
  
  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn
  ))
}


# dat <- datasets_classification_binary$banana
# x <- dat[,1:2]
# asd <- mice::ampute(data = x, prop = 0.5)
# x_missing <- asd$amp
# y <- dat[,3]
# 
# m_deepSMOTE <- resamplerCls_deepMIMSMOTE(x = x_missing, y = y, 
#                                          batch_size = 64L, epochs = 100L)
# plot(x, col = y)
# 
# plot(m_deepSMOTE$x_new, col = m_deepSMOTE$y_new)
# points(x = m_deepSMOTE$x_syn, col = "green")

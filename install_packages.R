# Installs the R packages used by the scripts. Run once:  Rscript install_packages.R
cran <- c("R6", "reticulate", "ggplot2", "openxlsx", "readxl", "dplyr", "tidyr", "reshape2", "caret", "yardstick",
          "kernlab", "e1071", "randomForest", "ranger", "rpart", "MASS", "mlr", "crayon", "parallel",
          "PMCMRplus", "ggsci", "ggpubr", "ggnewscale", "ggtext", "ggsignif", "patchwork", "cowplot", "bayesplot",
          # only for apps/shiny_app.R
          "shiny", "shinydashboard", "shinydashboardPlus", "shinyWidgets", "shinyEffects", "shinyAce", "DT", "reactable", "fresh")
missing <- setdiff(cran, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
# packages that are not on CRAN (or not in the version used here): install from GitHub if the next lines fail
if (!requireNamespace("SMOTEWB", quietly = TRUE)) install.packages("SMOTEWB")
if (!requireNamespace("bbtcomp", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("jwainer/bbtcomp")
}
# Python side
# reticulate::py_install(c("torch", "numpy", "scipy", "scikit-learn"))

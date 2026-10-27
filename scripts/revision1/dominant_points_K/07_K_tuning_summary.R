# ==============================================================================
# Script: scripts/revision1/dominant_points_K/07_K_tuning_summary.R   (NEW FILE; run from the project root, after "scripts/revision1/dominant_points_K/06_run_K_tuning_alpha.R")
# Purpose: appendix analysis "K tuned through alpha": how often was every alpha chosen, and how does the K-tuned run
#          compare with the CoverNET column of the paper (same 72 datasets, wins / ties / losses, paired Wilcoxon).
#          The paper's column stays the main result; this is an additional analysis. No raw metric values are averaged.
# Input : results/CoverNET K tuning - revision 1/results.xlsx (Means, BestParams), results/manuscript tables.xlsm (Combined)
# Output (results/CoverNET K tuning - revision 1/): k tuning vs paper - revision 1.csv, k tuning chosen alpha - revision 1.csv
# ==============================================================================

library(openxlsx)

DIR  <- file.path("results", "CoverNET K tuning - revision 1")
new  <- read.xlsx(file.path(DIR, "results.xlsx"), sheet = "Means")
bp   <- read.xlsx(file.path(DIR, "results.xlsx"), sheet = "BestParams")
old  <- read.xlsx("results/manuscript tables.xlsm", sheet = "Combined")
metrics <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", AUC = "ROCAUC")      # name in Combined = name in Means

vs_paper <- do.call(rbind, lapply(names(metrics), function(m) {
  d <- merge(old[old$Metric == m, c("Dataset", "CoverNET")],
             new[new$Metric == metrics[[m]], c("Dataset", "CCCDNNET")], by = "Dataset")
  data.frame(Metric = m, `N datasets` = nrow(d),
             `K tuned wins`   = sum(d$CCCDNNET >  d$CoverNET),
             `Ties`           = sum(d$CCCDNNET == d$CoverNET),
             `K tuned losses` = sum(d$CCCDNNET <  d$CoverNET),
             `Wilcoxon p`     = wilcox.test(d$CCCDNNET, d$CoverNET, paired = TRUE, exact = FALSE)$p.value,
             check.names = FALSE)
}))

bp$alpha    <- as.numeric(sub(".*alpha=([0-9.]+).*", "\\1", bp$BestParams))
alpha_df    <- as.data.frame(table(alpha = bp$alpha), responseName = "N datasets")

write.csv(vs_paper, file.path(DIR, "k tuning vs paper - revision 1.csv"),     row.names = FALSE)
write.csv(alpha_df, file.path(DIR, "k tuning chosen alpha - revision 1.csv"), row.names = FALSE)
print(vs_paper, digits = 3)
print(alpha_df)

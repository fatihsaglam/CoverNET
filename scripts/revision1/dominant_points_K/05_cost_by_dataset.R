# ==============================================================================
# Script: scripts/revision1/dominant_points_K/05_cost_by_dataset.R   (NEW; run from the project root, after "scripts/revision1/dominant_points_K/04_run_sensitivity_real_data.R")
# Purpose: Which K did every rule actually use, and how long did it take, by dataset and by dataset size (n, p)?
#          Answers: "K = 10 is faster - on which n and p, and what K did the CoverNet rule use there?"
# Input : results/K sensitivity/k sensitivity raw - revision 1.csv   (Kpos / Kneg / Runtime per fold)
#         results/manuscript tables.xlsm, sheet Combined             (n, p, IR of the 72 datasets)
# Output (results/K sensitivity/): k sensitivity cost by dataset - revision 1.csv,
#         k sensitivity K distribution - revision 1.csv, k sensitivity cost by size - revision 1.csv
# No torch needed. Runtime and K are not performance metrics; medians and per-dataset ratios are used anyway.
# ==============================================================================

library(openxlsx)

DIR <- file.path("results", "K sensitivity")
raw <- read.csv(file.path(DIR, "k sensitivity raw - revision 1.csv"), check.names = FALSE, stringsAsFactors = FALSE)

comb <- read.xlsx("results/manuscript tables.xlsm", sheet = "Combined")
ds_chars <- unique(comb[, c("Dataset", "n", "p", "IR")])

rules <- c("CoverNet", "Equal K", "Log K", "K = 1", "K = 5", "K = 10", "K = 20")
rules <- rules[rules %in% raw$Rule]

# ------------------------------------------------------------------------------
# 1. per dataset x rule: mean K and runtime over folds and seeds
# ------------------------------------------------------------------------------
agg <- aggregate(raw[, c("Kpos", "Kneg", "Runtime")], by = list(Dataset = raw$Dataset, Rule = raw$Rule),
                 FUN = mean, na.rm = TRUE)

# wide table: one row per dataset
wide <- ds_chars[ds_chars$Dataset %in% agg$Dataset, ]
for (r in rules) {
  a <- agg[agg$Rule == r, ]
  wide[[paste("Kpos", r)]]    <- a$Kpos[match(wide$Dataset, a$Dataset)]
  wide[[paste("Kneg", r)]]    <- a$Kneg[match(wide$Dataset, a$Dataset)]
  wide[[paste("Runtime", r)]] <- a$Runtime[match(wide$Dataset, a$Dataset)]
}
for (r in setdiff(rules, "CoverNet"))
  wide[[paste("Runtime ratio", r, "to CoverNet")]] <- wide[[paste("Runtime", r)]] / wide[["Runtime CoverNet"]]
wide <- wide[order(wide$n, wide$p), ]
write.csv(wide, file.path(DIR, "k sensitivity cost by dataset - revision 1.csv"), row.names = FALSE)

# ------------------------------------------------------------------------------
# 2. distribution of the K actually used (over the 72 datasets), every rule
# ------------------------------------------------------------------------------
qfun <- function(x) c(Min = min(x), Q1 = unname(quantile(x, .25)), Median = median(x),
                      Q3 = unname(quantile(x, .75)), Max = max(x), Mean = mean(x))
kd <- do.call(rbind, lapply(rules, function(r) {
  a <- agg[agg$Rule == r, ]
  rbind(data.frame(Rule = r, Class = "minority", t(qfun(a$Kpos)), check.names = FALSE),
        data.frame(Rule = r, Class = "majority", t(qfun(a$Kneg)), check.names = FALSE))
}))
write.csv(kd, file.path(DIR, "k sensitivity K distribution - revision 1.csv"), row.names = FALSE)
print(kd, digits = 3)

# ------------------------------------------------------------------------------
# 3. runtime and K by size groups (quartiles of n and of p, like the figures in the paper)
# ------------------------------------------------------------------------------
bin_quartiles <- function(x) {
  qs <- unique(quantile(x, probs = c(0, .25, .5, .75, 1), na.rm = TRUE))
  lab <- sprintf("[%g, %g]", round(qs[-length(qs)]), round(qs[-1]))
  cut(x, breaks = qs, labels = lab, include.lowest = TRUE)
}
wide$`n group` <- bin_quartiles(wide$n)
wide$`p group` <- bin_quartiles(wide$p)

size_rows <- list()
for (f in c("n group", "p group")) {
  for (g in levels(wide[[f]])) {
    ii <- which(wide[[f]] == g)
    if (!length(ii)) next
    for (r in rules) {
      size_rows[[length(size_rows) + 1]] <- data.frame(
        Factor = f, Group = g, `N datasets` = length(ii), Rule = r,
        `Median Kpos` = median(wide[[paste("Kpos", r)]][ii]),
        `Median Kneg` = median(wide[[paste("Kneg", r)]][ii]),
        `Median runtime (s)` = median(wide[[paste("Runtime", r)]][ii]),
        `Median runtime ratio to CoverNet` =
          if (r == "CoverNet") 1 else median(wide[[paste("Runtime ratio", r, "to CoverNet")]][ii]),
        check.names = FALSE, stringsAsFactors = FALSE)
    }
  }
}
size_df <- do.call(rbind, size_rows)
write.csv(size_df, file.path(DIR, "k sensitivity cost by size - revision 1.csv"), row.names = FALSE)
print(size_df, digits = 3)

cat("\nDone.\n")

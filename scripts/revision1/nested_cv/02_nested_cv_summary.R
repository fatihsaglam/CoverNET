# ==============================================================================
# Script: scripts/revision1/nested_cv/02_nested_cv_summary.R   (NEW FILE, put it next to run.R)
# Purpose: statistics of the nested CV run (reviewer 3, comments 4 and 5). Never averages raw metrics over datasets:
#          ranks, W/T/L and Holm adjusted tests only (paired Wilcoxon, normal approximation; Conover post hoc Friedman).
# Scenarios (higher is better for every metric):
#   paper flat                : the 19 methods of the paper, flat CV (reproduces the paper)
#   paper methods nested      : the same 19 methods, tuned neural methods under nested CV, untuned methods unchanged
#   tuned RF and SVM nested   : RF, RUS RF, SMOTE RF, SVM, SMOTE SVM replaced by their tuned nested versions
# Input : results/nested CV - revision 1/results.xlsx, results/results combined.xlsx
# Output (results/nested CV - revision 1/):
#   nested CV mean ranks - revision 1.csv        mean rank of every method, per scenario and metric
#   nested CV CoverNET vs rivals - revision 1.csv  W/T/L, rank difference, Holm Wilcoxon, Holm Conover
#   nested CV nested vs flat - revision 1.csv    per method: datasets where nested is lower / tied / higher, Wilcoxon p
#                                                (tuned neural methods vs flat CV, tuned RF and SVM vs untuned default)
# Bayesian Bradley-Terry comparison (bbtcomp, same analysis and figure as 'statistical tests and plots.R') for the two
# nested scenarios, at the end of the script. Output:
#   results/Figures/Bayesian comparison boxplot <scenario> - revision 1.png
#   results/Bradley Terry model results <scenario> - revision 1.xlsx
# ==============================================================================

library(openxlsx)

NESTED  <- "results/nested CV - revision 1/results.xlsx"
PAPER   <- "results/results combined.xlsx"
OUT_DIR <- "results/nested CV - revision 1"
METRICS <- c("IAM", "MCC", "P4", "ROCAUC")

nested <- read.xlsx(NESTED, sheet = "Means")
paper  <- read.xlsx(PAPER,  sheet = "Means")
names(nested)[names(nested) == "CCCDNNET"] <- "CoverNET"
ds <- unique(nested$Dataset)

tuned_nn  <- c("NNET", "BalancedNNET", "SMOTENNET", "FocalNNET", "FocalSDNNET", "LDAMNNET", "MMNNET", "PNN", "SkewPNN", "CoverNET")
classical <- c("RF", "RUS_RF", "SMOTE_RF", "SVMRadial", "SMOTE_SVM")
tuned_cl  <- c("RFtuned", "RUSRFtuned", "SMOTERFtuned", "SVMtuned", "SMOTESVMtuned")     # same order as classical
other     <- c("NaiveBayes", "CART", "KNN", "LogReg")

# performance matrix (datasets x methods); a method is taken from the nested run if it is there, else from the paper
get_perf <- function(metric, methods, from_nested = TRUE, nst = nested, ppr = paper) {
  a <- nst[nst$Metric == metric, ]; a <- a[match(ds, a$Dataset), ]
  b <- ppr[ppr$Metric == metric, ]; b <- b[match(ds, b$Dataset), ]
  x <- sapply(methods, function(m) if (from_nested && m %in% names(a)) a[[m]] else b[[m]])
  matrix(x, nrow = length(ds), dimnames = list(ds, methods))
}

wilcox_p <- function(a, b) if (all(a == b)) 1 else wilcox.test(a, b, paired = TRUE, exact = FALSE, correct = FALSE)$p.value

# ranks, Holm adjusted Wilcoxon and Conover p values of every pair
analyse <- function(x) {
  r  <- t(apply(-x, 1, rank)); b <- nrow(x); k <- ncol(x); Rs <- colSums(r)
  S  <- sqrt(2 * b * (sum(r^2) - b * k * (k + 1)^2 / 4) / ((b - 1) * (k - 1)))
  pr <- t(combn(colnames(x), 2))
  list(mean_rank = colMeans(r),
       pairs = data.frame(a = pr[, 1], b = pr[, 2],
                          holm_wilcoxon = p.adjust(apply(pr, 1, function(p) wilcox_p(x[, p[1]], x[, p[2]])), "holm"),
                          holm_conover  = p.adjust(2 * pt(-abs(Rs[pr[, 1]] - Rs[pr[, 2]]) / S, (b - 1) * (k - 1)), "holm")))
}

scenarios <- list("paper flat"              = list(c(classical, other, tuned_nn), FALSE),
                  "paper methods nested"    = list(c(classical, other, tuned_nn), TRUE),
                  "tuned RF and SVM nested" = list(c(tuned_cl,  other, tuned_nn), TRUE))

ranks <- rivals <- list()
for (sc in names(scenarios)) for (mt in METRICS) {
  x   <- get_perf(mt, scenarios[[sc]][[1]], scenarios[[sc]][[2]])
  res <- analyse(x)
  ranks[[length(ranks) + 1]] <- data.frame(Scenario = sc, Metric = mt, Method = colnames(x), MeanRank = round(res$mean_rank, 3))
  p  <- res$pairs[res$pairs$a == "CoverNET" | res$pairs$b == "CoverNET", ]
  rv <- ifelse(p$a == "CoverNET", p$b, p$a)
  rivals[[length(rivals) + 1]] <- data.frame(
    Scenario = sc, Metric = mt, Rival = rv,
    Win  = sapply(rv, function(m) sum(x[, "CoverNET"] > x[, m])),
    Tie  = sapply(rv, function(m) sum(x[, "CoverNET"] == x[, m])),
    Loss = sapply(rv, function(m) sum(x[, "CoverNET"] < x[, m])),
    RankDiff = round(res$mean_rank[rv] - res$mean_rank["CoverNET"], 3),
    HolmWilcoxon = p$holm_wilcoxon, HolmConover = p$holm_conover)
}
ranks  <- do.call(rbind, ranks)
rivals <- do.call(rbind, rivals)

# nested vs flat (tuned neural methods) and tuned vs untuned default (RF and SVM families)
cmp <- data.frame(method = c(tuned_nn, tuned_cl), reference = c(tuned_nn, classical), stringsAsFactors = FALSE)
vs_flat <- do.call(rbind, lapply(METRICS, function(mt) do.call(rbind, lapply(seq_len(nrow(cmp)), function(i) {
  a <- get_perf(mt, cmp$method[i])[, 1]; b <- get_perf(mt, cmp$reference[i], FALSE)[, 1]
  data.frame(Metric = mt, Method = cmp$method[i], Reference = if (i <= length(tuned_nn)) "flat CV" else "untuned default",
             NestedLower = sum(a < b), Tie = sum(a == b), NestedHigher = sum(a > b), WilcoxonP = wilcox_p(a, b))
}))))

write.csv(ranks,   file.path(OUT_DIR, "nested CV mean ranks - revision 1.csv"),          row.names = FALSE)
write.csv(rivals,  file.path(OUT_DIR, "nested CV CoverNET vs rivals - revision 1.csv"),  row.names = FALSE)
write.csv(vs_flat, file.path(OUT_DIR, "nested CV nested vs flat - revision 1.csv"),      row.names = FALSE)

# on screen: CoverNET position and number of rivals it beats significantly
for (sc in names(scenarios)) for (mt in METRICS) {
  r <- ranks[ranks$Scenario == sc & ranks$Metric == mt, ]; v <- rivals[rivals$Scenario == sc & rivals$Metric == mt, ]
  cat(sprintf("%-24s %-6s CoverNET mean rank %.2f (position %d of %d) | Holm Wilcoxon < 0.05: %d of %d | Holm Conover < 0.05: %d of %d\n",
              sc, mt, r$MeanRank[r$Method == "CoverNET"], rank(r$MeanRank, ties.method = "min")[r$Method == "CoverNET"], nrow(r),
              sum(v$HolmWilcoxon < 0.05), nrow(v), sum(v$HolmConover < 0.05), nrow(v)))
}

# ------------------------------------------------------------------------------
# Bayesian Bradley-Terry comparison (bbtcomp): posterior probability that CoverNET beats every rival.
# Pairs of get_pwin are named "A > B" in either order, so every posterior is turned into P(CoverNET beats rival).
# Box colour from the posterior median and the ROPE (0.45, 0.55): red = CoverNET loses, orange = practically equal, green = wins.
# ------------------------------------------------------------------------------
library(bbtcomp)
library(ggplot2)
library(ggtext)
dir.create("results/Figures", showWarnings = FALSE)

nested_sd <- read.xlsx(NESTED, sheet = "SDs")
names(nested_sd)[names(nested_sd) == "CCCDNNET"] <- "CoverNET"
paper_sd  <- read.xlsx(PAPER, sheet = "SDs")

label <- c(RF = "RF", NaiveBayes = "NB", CART = "DT", SVMRadial = "SVM", KNN = "KNN", LogReg = "LogReg",
           SMOTE_RF = "SMOTE + RF", RUS_RF = "RUS + RF", SMOTE_SVM = "SMOTE + SVM", NNET = "NN", BalancedNNET = "BalNN",
           SMOTENNET = "SMOTE + NN", FocalNNET = "FocalNN", FocalSDNNET = "FocalSDNN", LDAMNNET = "LDAMNN", MMNNET = "MMNN",
           PNN = "PNN", SkewPNN = "SkewPNN", CoverNET = "CoverNET", RFtuned = "Tuned RF", RUSRFtuned = "Tuned RUS + RF",
           SMOTERFtuned = "Tuned SMOTE + RF", SVMtuned = "Tuned SVM", SMOTESVMtuned = "Tuned SMOTE + SVM")
tag      <- c("paper methods nested" = "nested paper methods", "tuned RF and SVM nested" = "nested tuned RF and SVM")
mlab     <- c(IAM = "IAM", MCC = "MCC", P4 = "P4", ROCAUC = "AUC")
ROPE     <- c(0.45, 0.55)
fill_map <- c("CoverNET loses" = "red", "Practically equal" = "orange", "CoverNET wins" = "darkgreen")

for (sc in names(tag)) {
  methods <- scenarios[[sc]][[1]]
  fits <- lapply(setNames(METRICS, METRICS), function(mt) {
    # bbtcomp reads the first column of tabx as the dataset names (without it the first method is dropped);
    # tabsd must be numeric only (the error "sdx[, i]^2 : non-numeric argument" comes from a name column in tabsd)
    x  <- data.frame(Dataset = ds, get_perf(mt, methods), check.names = FALSE); colnames(x) <- c("Dataset", label[methods])
    sd <- as.data.frame(get_perf(mt, methods, nst = nested_sd, ppr = paper_sd));  colnames(sd) <- label[methods]
    bbtcomp(tabx = x, tabsd = sd)
  })

  # method names as they may appear in the pair names (original or make.names version) -> display label
  lookup <- c(setNames(label[methods], label[methods]), setNames(label[methods], make.names(label[methods])))
  draws <- do.call(rbind, lapply(METRICS, function(mt) {
    pw <- as.data.frame(bbtcomp:::get_pwin(fits[[mt]]))
    pw <- pw[, grepl("CoverNET", colnames(pw), ignore.case = TRUE), drop = FALSE]
    do.call(rbind, lapply(colnames(pw), function(cn) {
      side <- unname(lookup[strsplit(cn, "\\s*>\\s*")[[1]]])                    # the two methods of the pair
      data.frame(Metric = mlab[[mt]], Rival = side[side != "CoverNET"],
                 Win = if (side[1] == "CoverNET") pw[[cn]] else 1 - pw[[cn]])
    }))
  }))

  if (length(unique(draws$Rival)) != length(methods) - 1) stop("bbtcomp lost a method, check the columns of tabx")

  d <- do.call(rbind, lapply(split(draws, list(draws$Metric, draws$Rival), drop = TRUE), function(g)
    data.frame(Metric = g$Metric[1], Rival = g$Rival[1], ymin = min(g$Win), lower = unname(quantile(g$Win, 0.025)),
               middle = median(g$Win), upper = unname(quantile(g$Win, 0.975)), ymax = max(g$Win))))
  d$result <- factor(ifelse(d$middle > ROPE[2], "CoverNET wins", ifelse(d$middle < ROPE[1], "CoverNET loses", "Practically equal")),
                     levels = names(fill_map))
  d$Metric   <- factor(d$Metric, levels = mlab)
  d          <- d[order(d$Metric, -d$middle), ]
  d$order_id <- ave(d$middle, d$Metric, FUN = seq_along)
  d$x_key    <- factor(paste(d$Metric, d$order_id, sep = "___"), levels = paste(d$Metric, d$order_id, sep = "___"))
  label_lookup <- setNames(paste0("<span style='color:", fill_map[as.character(d$result)], "'>", d$Rival, "</span>"), d$x_key)

  plt <- ggplot(d, aes(x = x_key)) +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), linewidth = 0.25, width = 0.5) +
    geom_boxplot(aes(ymin = ymin, lower = lower, middle = middle, upper = upper, ymax = ymax, fill = result),
                 stat = "identity", linewidth = 0.25) +
    scale_fill_manual(name = NULL, values = fill_map, limits = names(fill_map)) +
    geom_hline(yintercept = 0.5, color = "red", linetype = 1) +
    geom_hline(yintercept = ROPE, color = "red", linetype = 2) +
    facet_wrap(~ Metric, nrow = 1, scales = "free_x") +
    scale_x_discrete(name = "", labels = label_lookup) +
    scale_y_continuous(name = "Posterior win probability of CoverNET", breaks = seq(0, 1, 0.1)) +
    theme_bw() +
    theme(axis.text.x = element_markdown(angle = 90, hjust = 1, vjust = 0.5),
          legend.position = "bottom",
          strip.background = element_rect(fill = NA, color = NA))
  ggsave(sprintf("results/Figures/Bayesian comparison boxplot %s - revision 1.png", tag[[sc]]), plot = plt,
         width = 16, height = 6, dpi = 300)

  prob <- do.call(rbind, lapply(METRICS, function(mt) data.frame(Metric = mlab[[mt]], table_pwin(fits[[mt]], control = "CoverNET"))))
  win  <- do.call(rbind, lapply(METRICS, function(mt) data.frame(Metric = mlab[[mt]], table_wintable(fits[[mt]]$wintable, which = "both"))))
  write.xlsx(list(probTable = prob,
                  winTable  = win[win$alg1 %in% "CoverNET" | win$alg2 %in% "CoverNET", ],
                  posterior = d[, c("Metric", "Rival", "ymin", "lower", "middle", "upper", "ymax", "result")]),
             sprintf("results/Bradley Terry model results %s - revision 1.xlsx", tag[[sc]]))
}

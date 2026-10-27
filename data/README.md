# Datasets

`datasets_binary.rds` is a list of 72 data frames (one per binary imbalanced dataset; features first, class label in the last column; the minority class is the positive class). The 72 datasets derive from 30 source datasets ("families"); datasets of the same family share their features and are not independent.

**Preprocessing before storage.** Categorical features, where present, were converted to dummy (indicator) variables, and observations with missing values were removed. The source, URL/reference and license columns below are to be completed.

| Dataset | Family | n | p | IR | Source (UCI / KEEL / other) | URL or reference | License |
|---|---|---|---|---|---|---|---|
| BEED_epilepsy_0_vs_rest | BEED | 8000 | 16 | 3.00 |  |  |  |
| BEED_epilepsy_2_vs_rest | BEED | 8000 | 16 | 3.00 |  |  |  |
| BEED_epilepsy_3_vs_rest | BEED | 8000 | 16 | 3.00 |  |  |  |
| Fishing4 | Fishing4 | 1182 | 5 | 1.62 |  |  |  |
| Ionosphere | Ionosphere | 351 | 33 | 1.79 |  |  |  |
| Iranian_Customer_Churn | Iranian_Customer_Churn | 3150 | 13 | 5.36 |  |  |  |
| Raisin_Dataset | Raisin_Dataset | 900 | 7 | 1.00 |  |  |  |
| Sonar | Sonar | 208 | 60 | 1.14 |  |  |  |
| Vowel1 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel10 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel2 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel3 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel4 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel5 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel6 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel7 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel8 | vowel | 990 | 9 | 10.00 |  |  |  |
| Vowel9 | vowel | 990 | 9 | 10.00 |  |  |  |
| abalone19 | abalone | 4174 | 9 | 129.44 |  |  |  |
| abalone9_18 | abalone | 731 | 9 | 16.40 |  |  |  |
| abalone_21_vs_8 | abalone | 581 | 9 | 40.50 |  |  |  |
| amphibians_treeFrog | amphibians_treeFrog | 189 | 14 | 1.66 |  |  |  |
| banana | banana | 2640 | 2 | 9.00 |  |  |  |
| banknote | banknote | 200 | 6 | 1.00 |  |  |  |
| car_good | car_good | 1728 | 15 | 24.04 |  |  |  |
| care_home_incidents | care_home_incidents | 1216 | 11 | 1.38 |  |  |  |
| churn | churn | 5000 | 19 | 6.07 |  |  |  |
| dhfr | dhfr | 325 | 228 | 1.66 |  |  |  |
| ecoli1 | ecoli | 336 | 7 | 3.36 |  |  |  |
| ecoli2 | ecoli | 336 | 7 | 5.46 |  |  |  |
| ecoli3 | ecoli | 336 | 7 | 8.60 |  |  |  |
| ecoli_0_1_4_7_vs_5_6 | ecoli | 332 | 6 | 12.28 |  |  |  |
| ecoli_0_2_6_7_vs_3_5 | ecoli | 224 | 7 | 9.18 |  |  |  |
| ecoli_0_3_4_6_vs_5 | ecoli | 205 | 7 | 9.25 |  |  |  |
| ecoli_0_4_6_vs_5 | ecoli | 203 | 6 | 9.15 |  |  |  |
| ecoli_0_6_7_vs_3_5 | ecoli | 222 | 7 | 9.09 |  |  |  |
| glass6 | glass | 214 | 9 | 6.38 |  |  |  |
| glass_0_1_6_vs_2 | glass | 192 | 9 | 10.29 |  |  |  |
| kddcup_land_vs_portsweep | kddcup | 1061 | 63 | 49.52 |  |  |  |
| kddcup_land_vs_satan | kddcup | 1610 | 50 | 75.67 |  |  |  |
| kr_vs_k_one_vs_fifteen | kr_vs_k | 2244 | 20 | 27.77 |  |  |  |
| kr_vs_k_three_vs_eleven | kr_vs_k | 2935 | 20 | 35.23 |  |  |  |
| kr_vs_k_zero_vs_eight | kr_vs_k | 1460 | 20 | 53.07 |  |  |  |
| kr_vs_k_zero_vs_fifteen | kr_vs_k | 2193 | 20 | 80.22 |  |  |  |
| lungcancer | lungcancer | 241 | 3 | 1.39 |  |  |  |
| mfeatmor01 | mfeat | 2000 | 6 | 4.00 |  |  |  |
| mfeatmor1 | mfeat | 2000 | 6 | 9.00 |  |  |  |
| mfeatmor14 | mfeat | 2000 | 6 | 4.00 |  |  |  |
| page_blocks_1_3_vs_4 | page_blocks_1_3_vs_4 | 472 | 10 | 15.86 |  |  |  |
| poker_8_9_vs_6 | poker | 1485 | 10 | 58.40 |  |  |  |
| poker_8_vs_6 | poker | 1477 | 10 | 85.88 |  |  |  |
| psa | psa | 683 | 4 | 1.98 |  |  |  |
| segment0 | segment | 2308 | 19 | 6.02 |  |  |  |
| segment_brickface_vs_rest | segment | 2310 | 18 | 6.00 |  |  |  |
| segment_cement_vs_rest | segment | 2310 | 18 | 6.00 |  |  |  |
| segment_foliage_vs_rest | segment | 2310 | 18 | 6.00 |  |  |  |
| segment_path_vs_rest | segment | 2310 | 18 | 6.00 |  |  |  |
| segment_sky_vs_rest | segment | 2310 | 18 | 6.00 |  |  |  |
| shuttle_2_vs_5 | shuttle_2_vs_5 | 3316 | 9 | 66.67 |  |  |  |
| tuandromd | tuandromd | 4464 | 242 | 3.97 |  |  |  |
| vehicle0 | vehicle | 846 | 18 | 3.25 |  |  |  |
| vehicle2 | vehicle | 846 | 18 | 2.88 |  |  |  |
| vehicle3 | vehicle | 846 | 18 | 2.99 |  |  |  |
| vowel0 | vowel | 988 | 13 | 9.98 |  |  |  |
| winequality_red_3_9_vs_5 | winequality | 1482 | 11 | 58.28 |  |  |  |
| winequality_red_4 | winequality | 1599 | 11 | 29.17 |  |  |  |
| winequality_red_8_vs_6 | winequality | 656 | 11 | 35.44 |  |  |  |
| yeast5 | yeast | 1484 | 7 | 32.73 |  |  |  |
| yeast6 | yeast | 1484 | 7 | 41.40 |  |  |  |
| yeast8 | yeast | 1484 | 7 | 73.20 |  |  |  |
| yeast_1_vs_7 | yeast | 459 | 7 | 14.30 |  |  |  |
| yeast_2_vs_8 | yeast | 514 | 8 | 9.08 |  |  |  |

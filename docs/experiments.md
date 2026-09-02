# Experiment Summary

This document supplements the concise README. Values come from canonical result tables. Stage 1 ensemble rows are prediction-level ensembles; Stage 2 mean±std rows are independent-seed robustness statistics.

## Stage 1 ablations

The D128 MASTER ablations use the same seven-target prediction-level ensemble contract as the Stage 1 main result.

| Ablation | IC | RankIC | Top 5% (bp) | Top 10% (bp) |
|---|---:|---:|---:|---:|
| No market guidance | 0.0682 | 0.0994 | 56.10 | 51.01 |
| No intra-stock aggregation | 0.0677 | 0.1022 | 57.72 | 52.79 |
| No inter-stock aggregation | 0.0658 | 0.0973 | 62.65 | 53.64 |
| No temporal aggregation | 0.0683 | 0.1000 | 61.83 | 53.80 |

Relative to the full D128 MSE row (`IC 0.0688`, `RankIC 0.1004`, `Top 5% 59.29 bp`, `Top 10% 53.70 bp`), the ablation picture is metric-dependent. Removing inter-stock aggregation reduced IC and RankIC but increased the Top 5% return; removing intra-stock aggregation slightly increased RankIC while reducing the other displayed metrics. The evidence therefore does **not** support the claim that every component improves every metric. Market guidance and temporal aggregation show modest rather than uniformly dominant differences in this table.

## Stage 1 loss experiments

| Loss (D128) | IC | RankIC | Top 5% (bp) | Top 10% (bp) |
|---|---:|---:|---:|---:|
| MSE | 0.0688 | 0.1004 | 59.29 | 53.70 |
| CCC | 0.0726 | 0.1055 | 60.83 | 55.52 |
| ListFold | 0.0667 | 0.1120 | 48.29 | 45.56 |
| RPL | 0.0694 | 0.1015 | 57.54 | 51.68 |
| RSR | 0.0354 | 0.0764 | 33.94 | 30.76 |

CCC was strongest across the displayed D128 metrics; ListFold produced the highest RankIC but weaker top-ranked returns. RPL was close to MSE on correlations, while RSR was weaker in this experiment. All five formal loss families remain visible rather than selecting only the best row.

## Hyperparameters and training protocol

The historical screens cover:

| Dimension | Canonical values represented |
|---|---|
| Hidden dimension | 128, 256, 512 |
| Market-gate beta | 1, 2, 5, 10, 20, 40 |
| Dropout | 0.3, 0.4, 0.5, 0.7 |
| Learning rate | `3e-6`, `1e-5`, `3e-5` |
| Temporal/spatial heads | multiple 1/2/4-head combinations in Stage 1; explicit Stage 2 variants below |
| Rolling frequency | semiannual, quarterly, monthly |
| Initialization | warm, cold |

Beta was a formal main screen. Multiple dropout screens did not show a stable improvement. Learning-rate variants in the standard daily-market setting did not provide a clear strong gain. For richer intraday-market/joint-gating models, runs reaching the epoch cap indicate that optimization remained under-explored; they do not prove that a different learning rate would improve results.

## Rolling frequency

The common-date D128 comparison records quarterly IC/Top 5% return above the semiannual row, while RankIC is slightly below and monthly is based on one complete historical experiment. The evidence does not establish a high-capacity interaction, so no claim is made that the quarterly advantage necessarily disappears as model capacity increases.

## Stage 2 Market63 versus Market84

All rows below are complete seed 42/43/44 robustness summaries.

| Market state | Capacity | IC | RankIC | Top 5% (bp) |
|---|---|---:|---:|---:|
| Market63 | D256 | 0.0796 ± 0.0002 | 0.0994 ± 0.0055 | 62.18 ± 2.60 |
| Market63 | D512 | 0.0779 ± 0.0042 | 0.1015 ± 0.0011 | 63.69 ± 5.68 |
| Market84 | D256 | 0.0800 ± 0.0020 | 0.1006 ± 0.0037 | 59.14 ± 4.57 |
| Market84 | D512 | 0.0761 ± 0.0017 | 0.1002 ± 0.0041 | 56.52 ± 5.90 |

Market84 made small mixed changes at D256 and was lower on the displayed D512 means. It did not produce a consistent improvement under these tested settings.

## Stage 2 attention heads

The five complete D512 settings are `T2×S2`, `T4×S1`, `T4×S2`, `T4×S4`, and `T8×S2`. RankIC and Top 5% return do not select the same setting, reinforcing the need to report multiple evaluation views rather than a single tuned number.

Selected 2025H2 / 2026H1_partial diagnostics for the README settings are available in [`late_regime_results.csv`](../results/stage2/late_regime_results.csv). The table retains the reported target-level Stage 2 metrics; the original raw target-evaluation artifacts are not included in this repository.

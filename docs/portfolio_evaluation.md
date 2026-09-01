# Portfolio Evaluation

Portfolio statistics are supplemental evidence from selected signals evaluated in the institutional backtest system used during the internship. The system evaluated daily holdings against stock returns; its precise execution-price convention and engine implementation are not part of this public repository. Transaction fees were not deducted. Because only a subset of signals entered this system, the rows below are not a controlled comparison across every prediction experiment.

The source called its selected long leg `Top`. Its exact selection rule is not recoverable from the available export, so the public presentation uses the neutral name **Long portfolio**. This is distinct from prediction-evaluator **Top 5% / Top 10%**, which explicitly means the top fraction of each daily stock cross-section ranked by model score.

## Selected runs

| Setting | IC | Long annual return | Long Sharpe | Long-short annual return | Long-short Sharpe | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM · Stage 1 · M63 · Seed42 | 0.057 | 10.6% | 0.684 | 89.7% | 5.128 | 0.411 |
| Stage 1 · M63 · D128 · H4×1 · Seeds42/43/44 Ensemble | 0.069 | 17.1% | 1.052 | 108.0% | 5.625 | 0.271 |
| Stage 2 · M63 · D256 · H4×2 · Seed43 | 0.070 | 18.3% | 1.119 | 108.3% | 6.108 | 0.376 |
| Stage 2 · M63 · D512 · H4×2 · Seed42 | 0.069 | 21.4% | 1.363 | 110.9% | 6.711 | 0.541 |

Among these representative backtests, the D512 intraday model has higher portfolio return and Sharpe, together with higher turnover. This is a descriptive observation about the displayed historical runs, not a causal feature-ablation claim.

## Full sanitized aggregate table

| Public setting | Period | Status | Long total return | Long annual return | Long Sharpe | Long-short total return | Long-short annual return | Long-short Sharpe | IC | ICIR | Turnover |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM · Stage 1 · M63 · Seed42 | 2023-01-03 – 2026-04-27 | reference baseline | 34.9% | 10.6% | 0.684 | 294.1% | 89.7% | 5.128 | 0.057 | 0.485 | 0.411 |
| Stage 1 · M63 · D128 · H4×1 · Seeds42/43/44 Ensemble | 2023-01-01 – 2026-04-27 | selected Stage 1 | 56.1% | 17.1% | 1.052 | 354.2% | 108.0% | 5.625 | 0.069 | 0.535 | 0.271 |
| MASTER D128 · no inter-stock attention | 2023-01-01 – 2026-04-27 | Stage 1 ablation | 57.2% | 17.4% | 1.236 | 347.2% | 105.9% | 5.846 | 0.068 | 0.605 | 0.297 |
| MASTER D128 · no intra-stock aggregation | 2023-01-01 – 2026-04-27 | Stage 1 ablation | 62.1% | 18.9% | 1.183 | 351.6% | 107.2% | 5.640 | 0.072 | 0.543 | 0.326 |
| MASTER D128 · no market guidance | 2023-01-01 – 2026-04-27 | Stage 1 ablation | 50.1% | 15.3% | 0.917 | 346.5% | 105.7% | 5.453 | 0.068 | 0.517 | 0.264 |
| MASTER D128 · no temporal aggregation | 2023-01-01 – 2026-04-27 | Stage 1 ablation | 56.9% | 17.4% | 1.152 | 342.5% | 104.5% | 5.548 | 0.068 | 0.538 | 0.360 |
| MASTER D128 · cold start | 2023-01-01 – 2026-04-27 | initialization comparison | 49.2% | 15.0% | 0.972 | 328.0% | 100.0% | 5.236 | 0.068 | 0.540 | 0.264 |
| MASTER D128 · warm start | 2023-01-03 – 2026-04-27 | initialization comparison | 56.2% | 17.2% | 1.060 | 343.1% | 104.6% | 5.450 | 0.066 | 0.487 | 0.302 |
| Stage 2 · M63 · D256 · H4×2 · Seed43 | 2023-01-01 – 2026-04-27 | selected Stage 2 single seed | 59.9% | 18.3% | 1.119 | 355.1% | 108.3% | 6.108 | 0.070 | 0.539 | 0.376 |
| Stage 2 · M63 · D512 · H4×2 · Seed42 | 2023-01-01 – 2026-04-27 | selected Stage 2 single seed | 70.3% | 21.4% | 1.363 | 363.6% | 110.9% | 6.711 | 0.069 | 0.545 | 0.541 |

The machine-readable version is [portfolio_results.csv](../results/portfolio/portfolio_results.csv). Date starts are source-reported boundaries: some exports begin at `2023-01-01`, while those with an explicit first trading date begin at `2023-01-03`.

The institutional exports also contained style-exposure diagnostics. They are not published because the available images do not provide a sufficiently self-contained public definition of the underlying factor model or structured data for interpretation.

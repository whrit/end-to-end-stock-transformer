# Dual-Frequency Market Guidance (Exploratory)

This secondary Stage 2 study combines daily Market84 and minute-level Intraday Market Features (192D) with the Alpha158 + Intraday Price-Volume Features (248D) stock representation. It tests separate matched routing and learned joint all-to-all routing, plus hidden-capacity and attention-head variants.

The table reports one historical seed per row. It is not a multi-seed ensemble or seed-robustness table.

| Public order | Routing | Capacity | Heads | Coverage | IC | RankIC | Top 5% (bp) | Top 10% (bp) |
|---|---|---:|---|---|---:|---:|---:|---:|
| exp001 | separate/matched | D128 | T4×S2 | complete (7/7) | 0.0756 | 0.0944 | 60.31 | 52.37 |
| exp002 | joint/all-to-all | D128 | T4×S2 | complete (7/7) | 0.0779 | 0.0939 | 64.24 | 55.46 |
| exp003 | separate/matched | D256 | T4×S2 | complete (7/7) | 0.0813 | 0.0968 | 65.41 | 58.15 |
| exp004 | joint/all-to-all | D256 | T4×S2 | complete (7/7) | 0.0836 | 0.0977 | 65.54 | 59.14 |
| exp005 | separate/matched | D512 | T4×S2 | complete (7/7) | 0.0799 | 0.1029 | 67.02 | 60.78 |
| exp006 | separate/matched | D128 | T2×S2 | complete (7/7) | 0.0765 | 0.0930 | 60.27 | 52.95 |
| exp007 | separate/matched | D128 | T8×S2 | partial (2/7) | 0.0718 | 0.0881 | — | — |
| exp008 | separate/matched | D128 | T4×S4 | partial (2/7) | 0.0715 | 0.0874 | — | — |

`exp001–exp008` is the sequential presentation order of the eight reported rows. The first six rows are complete, while exp007–008 retain partial IC/RankIC and missing Top 5% / Top 10% values.

Within this limited single-seed sample, joint routing is higher than separate routing for the matched D128 and D256 pairs on IC and top-ranked returns, while the best metric varies across capacity and heads. The study remained exploratory and was not exhaustively tuned. It was not selected as the main reported configuration, and optimization of the richer joint-gating setting remains under-explored.

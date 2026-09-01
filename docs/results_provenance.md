# Results Provenance

The canonical `results/` package normalizes recovered research evidence without changing historical metric values or pretending that unavailable artifacts exist.

## Canonical layers

- `experiment_registry.csv` identifies public settings, historical identities, coverage, aggregation, evidence level, and source descriptors.
- `stage1/` contains main, baseline, ablation, loss, rolling, and hyperparameter tables.
- `stage2/` contains complete/partial capacity, Market84, attention-head, dual-frequency, and seed-robustness tables.
- `portfolio/portfolio_results.csv` contains sanitized aggregate statistics for the subset of signals with recovered historical institutional backtest exports.
- `metadata/` contains the historical experiment and result-source registries.
- `metric_definitions.json` fixes metric units and aggregation semantics.
- `provenance/source_inventory_summary.json` preserves row counts, the deleted full-inventory hash, role/type summaries, and an aggregate Project04 evaluation-manifest hash.
- `provenance/project03_primary_source_hashes.json` records hashes for the selected primary Stage 1 recovery sources.
- `conflicts.csv` preserves known source disagreements instead of averaging or overwriting them.

## Stage 1 recovery and ensembles

Stage 1 tables were recovered from historical completed metric exports plus evaluator/aggregation evidence. A historical three-seed ensemble means: align seed 42/43/44 raw predictions on `trade_date × sec_code`, take an equal-weight arithmetic mean of `score`, and run the evaluator again. It never means averaging three metric values.

`ensemble_semantics_evidence` distinguishes:

- `source_proven` — a recovered averaging implementation directly proves prediction-level averaging followed by evaluation. This covers MASTER main, ablation, rolling, and D128 loss ensembles.
- `user_confirmed` — the project author confirms the same historical contract, but no missing script, hash, or raw artifact is fabricated. This covers GRU, Vanilla Transformer, and D512 CCC ensembles.

## Stage 2 recovery

Stage 2 overall metrics were reconstructed from materialized target evaluation summaries using valid-day weighting for IC/RankIC/Top returns and pooled observations for MSE/R². There was no historical materialized three-seed raw-prediction ensemble, so `historical_raw_prediction_ensemble=false`. Mean ± sample standard deviation rows are robustness summaries across independently trained seeds.

D256, D512, Market84, and the five head settings have complete seed 42/43/44 × 7/7 coverage. D128 is partial: seed 42 and 43 cover 7/7, seed 44 covers 1/7. Dual-frequency exp001–006 are complete 7/7 single-seed experiments; exp007–008 are partial 2/7.

No exact Stage 2 protocol-matched Alpha158-only control exists (`matched_control=false`). Cross-stage comparisons must not be described as a clean causal intraday-feature ablation.

## Portfolio evidence

Portfolio results are sanitized aggregate statistics recovered from historical institutional backtest exports. The private `backtesting/` source directory is excluded from Git, and the public repository does not contain or claim to reproduce the institutional portfolio engine. The aggregate table excludes fields whose public definitions are unavailable, uses **Long portfolio** for the source's otherwise unspecified selected long leg, and records that transaction fees were not deducted.

Only selected historical signals entered that backtest system. Portfolio rows are therefore supplemental research evidence rather than a matched result for every prediction experiment. The public method notes and complete sanitized aggregate table are in [portfolio evaluation](portfolio_evaluation.md).

## Conflicts and missing values

`conflicts.csv` retains 14 tiny IC/RankIC differences caused by different historical aggregation paths. They are on the order of roughly `1e-11` for IC and `1e-9` for RankIC. The selected primary source remains the basis for the Stage 1 main table; the alternative values are neither rounded away nor merged.

Missing Top 5% / Top 10% values for partial experiments remain `NaN`. Nine unresolved historical identities remain in metadata: one monthly setting, five baseline screening settings, and three Stage 2 Alpha158-only control settings. These are explicitly historical plan/config identities without recovered result evidence. Unexecuted dual-frequency tuning-matrix and auxiliary-ablation plan rows were removed before release.

All location strings in public provenance are sanitized relative/original-location descriptors; they are not runnable internal infrastructure paths.

The original 12,445-row recovery inventory was a one-time migration audit artifact, not a public user dependency. It was replaced by the compact integrity summary while retaining selected Stage 1 hashes, per-result source descriptors, experiment manifests, and `conflicts.csv`.

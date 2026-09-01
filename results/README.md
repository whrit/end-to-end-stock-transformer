# Results

This directory contains the normalized public research results and their sanitized historical evidence.

- `stage1/` — MASTER baseline, neural/tree baselines, ablations, losses, rolling comparisons, and seed robustness.
- `stage2/` — intraday-extension capacity, Market84, attention-head, dual-frequency, and seed-robustness results.
- `portfolio/` — selected portfolio evaluation aggregates.
- `metadata/` — historical experiment and result-source registries.
- `provenance/` — sanitized manifests, hashes, and recovery evidence.

Top-level files:

- `experiment_registry.csv` — public settings, historical identities, coverage, and aggregation semantics.
- `metric_definitions.json` — metric definitions and units.
- `conflicts.csv` — known small disagreements between historical aggregation sources.

See [`docs/experiments.md`](../docs/experiments.md), [`docs/results_provenance.md`](../docs/results_provenance.md), and [`docs/portfolio_evaluation.md`](../docs/portfolio_evaluation.md) for interpretation and limitations.

# Reproducibility

## Historical environment versus public validation

Historical experiments were run with **Python 3.8.13** in a restricted research compute environment using specialized accelerator hardware. Those constraints describe the original experiment environment, not a dependency of the public package.

The canonical codebase:

- removes environment-specific database and infrastructure dependencies from the public workflow;
- uses standard PyTorch device resolution for CPU/CUDA execution;
- has been audited for Python 3.8 grammar compatibility;
- separates immutable derived-data construction from repeated model training;
- includes a focused test suite spanning formulas, contracts, storage, datasets, models, training, rolling orchestration, evaluation, and configuration parsing.

Passing the public test suite validates the public code path, but is not the same as rerunning historical full-scale training on the original hardware. No unverified Python-version compatibility claim is made beyond the declared `Python >= 3.8` package requirement and the grammar audit.

## Reproduction levels

1. **Formula and contract reproduction** — install the package and run `pytest` without proprietary data.
2. **Pipeline reproduction with user data** — map daily/minute data, membership, and calendar to [the public contract](data_contract.md), then build immutable derived artifacts and run a config.
3. **Historical metric reproduction** — additionally requires equivalent raw market coverage, universe snapshots, adjustment inputs, and adequate compute. The original raw data cannot be redistributed.

## Determinism and artifacts

- Configs explicitly record model, optimizer, loss, rolling frequency, initialization, seed list, label contract, feature layout, and output root.
- Python, NumPy, and PyTorch RNGs are seeded; deterministic PyTorch behavior is requested by the training runner.
- Derived artifacts are immutable directories or files with manifests, dimensions, dtypes, date/code axes, provenance, and SHA-256 hashes where applicable.
- Checkpoints record model, optimizer, epoch, feature layout, seed, and config fingerprints. Resume verifies compatibility before skipping completed targets.
- A three-seed ensemble averages raw scores on identical `trade_date × sec_code` keys, then executes the evaluator on the new prediction. Seed mean/std is stored separately as robustness evidence.

## Suggested commands

```bash
pip install -e ".[test]"
pytest
```

Install optional dependencies only when needed: `.[lightgbm]`, `.[parquet]`, or `.[test]`.

## Known limits

- Public market data is not bundled and no vendor download claim is made.
- Historical result tables include complete and partial experiments; unavailable metrics are not imputed.
- Stage 2 does not contain a strictly matched Alpha158-only control under the same protocol.
- Historical Stage 2 tables contain independently evaluated seed results rather than a materialized raw-prediction ensemble.
- Portfolio aggregates are included in the public results, while the complete backtest implementation, execution settings, and underlying non-public evidence are not part of the repository.

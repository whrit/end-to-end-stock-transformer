"""Contracts for the normalized Stage 2 late-regime result table."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "stage2" / "late_regime_results.csv"
REQUIRED_COLUMNS = (
    "experiment",
    "model",
    "feature_stage",
    "market_state",
    "d_model",
    "temporal_heads",
    "spatial_heads",
    "seed",
    "target",
    "IC",
    "RankIC",
    "Top5_bp",
    "Top10_bp",
    "historical_experiment_id",
    "evidence_status",
)


def test_late_regime_result_contract():
    frame = pd.read_csv(CSV_PATH)

    assert tuple(frame.columns) == REQUIRED_COLUMNS
    assert len(frame) == 8
    assert frame["experiment"].nunique() == 4
    assert set(frame["target"]) == {"2025H2", "2026H1_partial"}
    assert frame.groupby("experiment")["target"].nunique().eq(2).all()
    assert not frame.duplicated(["experiment", "target"]).any()
    assert frame["evidence_status"].eq(
        "historical_target_metric_recovered"
    ).all()

from __future__ import annotations

import json
from pathlib import Path
from statistics import stdev

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


RESULT_FILES = [
    RESULTS / "stage1" / "main_results.csv",
    RESULTS / "stage1" / "baseline_results.csv",
    RESULTS / "stage1" / "ablation_results.csv",
    RESULTS / "stage1" / "loss_results.csv",
    RESULTS / "stage1" / "rolling_results.csv",
    RESULTS / "stage1" / "hyperparameter_results.csv",
    RESULTS / "stage2" / "main_results.csv",
    RESULTS / "stage2" / "capacity_results.csv",
    RESULTS / "stage2" / "market84_results.csv",
    RESULTS / "stage2" / "head_sensitivity.csv",
    RESULTS / "stage2" / "dual_results.csv",
]


def test_primary_recovered_source_hashes_are_pinned() -> None:
    manifest = json.loads(
        (RESULTS / "provenance" / "project03_primary_source_hashes.json").read_text(encoding="utf-8")
    )
    actual = {Path(row["source_file"]).name: row["sha256"] for row in manifest["sources"]}
    assert actual == {
        "collect_master_ppt_metrics_20260828.py": "c4f8cc8d4a021536f6c5e705d3088705a664fc539b626ccc06aa4ac4e3bb5ee3",
        "master_baseline_unified_metrics_20260828.csv": "7b42719ec25b8814cd006470212e83f9f5867d49595fc7a4dce8ef05f9685270",
        "master_ppt_completed_metrics_20260828.csv": "2f0d9755a26f04b45fd0163fb9799fe3da69bc1478862e846c8adc63b6effd40",
        "project03_existing_recovered_results_v2_20260831.csv": "dfe6650d31ddd8b167b1688ca7cab27f9bc694f6e259c6b8d3349bb421568615",
        "project03_seed_robustness_v2_20260831.csv": "6abbe7aba50fc30fa7b4e06c85e4237364dd2cea2ba951ad4f55241e30b86a80",
    }


@pytest.mark.parametrize("path", RESULT_FILES)
def test_no_duplicate_canonical_result_keys(path: Path) -> None:
    frame = pd.read_csv(path)
    keys = ["public_display_name", "seed", "target", "comparison_scope"]
    assert not frame.duplicated(keys, keep=False).any(), path


@pytest.mark.parametrize("path", RESULT_FILES)
def test_top_return_units_are_converted_once(path: Path) -> None:
    frame = pd.read_csv(path)
    for source, unit, bp in (("Top5_source", "Top5_source_unit", "Top5_bp"), ("Top10_source", "Top10_source_unit", "Top10_bp")):
        populated = frame.dropna(subset=[source])
        decimal = populated[populated[unit] == "decimal_return"]
        basis_points = populated[populated[unit] == "basis_points"]
        assert np.allclose(decimal[bp], decimal[source] * 10_000.0, rtol=0.0, atol=1e-10)
        assert np.allclose(basis_points[bp], basis_points[source], rtol=0.0, atol=1e-10)


def test_known_decimal_conversion_is_sixty_bp_not_double_converted() -> None:
    decimal = 0.006
    assert decimal * 10_000.0 == 60.0
    definitions = json.loads((RESULTS / "metric_definitions.json").read_text(encoding="utf-8"))
    assert "10,000" in definitions["canonical_columns"]["Top5_bp"]


@pytest.mark.parametrize("path", RESULT_FILES)
def test_incomplete_experiments_are_not_marked_complete(path: Path) -> None:
    frame = pd.read_csv(path)
    incomplete = frame[frame["targets_completed"] < frame["targets_expected"]]
    assert not incomplete["completion_status"].eq("complete").any()


def test_project04_seed_robustness_is_not_prediction_ensemble() -> None:
    frame = pd.read_csv(RESULTS / "stage2" / "seed_robustness.csv")
    assert frame["aggregation"].eq("seed_robustness_mean_std").all()
    assert not frame["aggregation"].str.contains("prediction_ensemble", case=False).any()

    capacity = pd.read_csv(RESULTS / "stage2" / "capacity_results.csv")
    d256 = capacity[(capacity["d_model"] == 256) & (capacity["completion_status"] == "complete")]
    row = frame[frame["public_display_name"] == "capacity_d256_market63_t4_s2_seed_robustness"].iloc[0]
    assert row["IC_std"] == pytest.approx(stdev(d256["IC"].tolist()))


def test_project03_raw_ensemble_distinguishes_source_and_user_evidence() -> None:
    registry = pd.read_csv(RESULTS / "experiment_registry.csv")
    p03 = registry[registry["project"] == "Project03"]
    ensembles = p03[p03["historical_raw_prediction_ensemble"] == True]  # noqa: E712
    assert not ensembles.empty
    assert set(ensembles["ensemble_semantics_evidence"]) == {"source_proven", "user_confirmed"}
    proven = ensembles[ensembles["ensemble_semantics_evidence"] == "source_proven"]
    assert proven["provenance_source"].notna().all()

    baseline = pd.read_csv(RESULTS / "stage1" / "baseline_results.csv")
    neural = baseline[baseline["model"].isin(["GRU", "VanillaTransformer"])]
    assert neural["historical_raw_prediction_ensemble"].all()
    assert neural["aggregation"].eq("raw_prediction_ensemble").all()
    assert neural["ensemble_semantics_evidence"].eq("user_confirmed").all()

    loss = pd.read_csv(RESULTS / "stage1" / "loss_results.csv")
    d512_ccc = loss[loss["public_display_name"] == "loss_ccc_d512"]
    assert d512_ccc["historical_raw_prediction_ensemble"].all()
    assert d512_ccc["ensemble_semantics_evidence"].eq("user_confirmed").all()

    source_proven = loss[(loss["seed"] == "ensemble") & (loss["d_model"] == 128)]
    assert source_proven["ensemble_semantics_evidence"].eq("source_proven").all()

    rolling = pd.read_csv(RESULTS / "stage1" / "rolling_results.csv")
    assert rolling["aggregation"].eq("raw_prediction_ensemble").all()
    assert rolling["historical_raw_prediction_ensemble"].all()
    assert rolling["ensemble_semantics_evidence"].eq("source_proven").all()
    assert rolling["provenance_source"].notna().all()


def test_missing_metrics_remain_missing() -> None:
    screening = pd.read_csv(RESULTS / "stage1" / "hyperparameter_results.csv")
    assert screening["Top5_bp"].isna().any()
    assert screening["Top10_bp"].isna().any()

    dual = pd.read_csv(RESULTS / "stage2" / "dual_results.csv")
    partial = dual[dual["public_display_name"].isin(["gsep_d128_th8_sh2", "gsep_d128_th4_sh4"])]
    assert partial["Top5_bp"].isna().all()
    assert partial["Top10_bp"].isna().all()


def test_public_result_csvs_contain_no_internal_absolute_paths() -> None:
    forbidden = ("/mnt/data01", "intern001", "\\mnt\\data01")
    for path in RESULTS.rglob("*.csv"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(item.lower() in text for item in forbidden), path


def test_metadata_is_reconciled_with_recovered_results() -> None:
    historical = pd.read_csv(RESULTS / "metadata" / "historical_experiments.csv")
    sources = pd.read_csv(RESULTS / "metadata" / "result_sources.csv")
    assert len(historical) == len(sources) == 99
    assert historical["public_display_name"].equals(sources["public_display_name"])
    assert historical["historical_execution_id"].fillna("").equals(sources["historical_execution_id"].fillna(""))
    assert historical["recovery_status"].equals(sources["recovery_status"])
    assert historical["recovery_status"].value_counts().to_dict() == {
        "recovered": 85, "unresolved": 9, "recovered_partial": 5,
    }
    recovered = sources["recovery_status"].str.startswith("recovered")
    assert sources.loc[recovered, "result_availability"].eq("available").all()
    assert sources.loc[recovered, "needs_cloud_export"].eq(False).all()  # noqa: E712
    assert sources.loc[~recovered, "result_availability"].eq("missing").all()
    assert sources.loc[~recovered, "needs_cloud_export"].eq(True).all()  # noqa: E712


def test_results_and_metadata_are_sanitized() -> None:
    forbidden = ("/mnt/data01", "intern001", "datayes", "sqlite", "password", "api_key")
    for folder in (RESULTS,):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".txt"}:
                text = path.read_text(encoding="utf-8").lower()
                assert not any(item in text for item in forbidden), path


def test_project04_never_claims_historical_prediction_ensemble() -> None:
    registry = pd.read_csv(RESULTS / "experiment_registry.csv")
    project04 = registry[registry["project"] == "Project04"]
    assert not project04["historical_raw_prediction_ensemble"].any()
    assert project04["ensemble_semantics_evidence"].isna().all()


def test_canonical_runtime_does_not_reference_extracted_recovery_directories() -> None:
    recovery_names = (
        "project03_" + "github_handoff_20260831",
        "project04_" + "results_export_20260831",
    )
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(name in text for name in recovery_names), path


def test_dual_partial_runs_are_excluded_from_complete_only_comparison() -> None:
    dual = pd.read_csv(RESULTS / "stage2" / "dual_results.csv")
    by_name = dual.set_index("public_display_name")
    assert by_name.loc["gsep_d128_th8_sh2", "targets_completed"] == 2
    assert by_name.loc["gsep_d128_th4_sh4", "targets_completed"] == 2
    assert by_name.loc["gsep_d128_th8_sh2", "completion_status"] == "partial"
    assert by_name.loc["gsep_d128_th4_sh4", "completion_status"] == "partial"
    assert dual[dual["completion_status"] == "complete"].shape[0] == 6


def test_stage2_matched_alpha158_only_control_is_not_claimed() -> None:
    registry = pd.read_csv(RESULTS / "experiment_registry.csv")
    control = registry[registry["family"] == "matched_control"]
    assert len(control) == 1
    assert control.iloc[0]["status"] == "not_found"
    assert not bool(control.iloc[0]["matched_control"])


def test_compact_recovery_inventory_summary_is_pinned() -> None:
    summary = json.loads(
        (RESULTS / "provenance" / "source_inventory_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["full_inventory"] == {
        "row_count": 12445,
        "size_bytes": 5155573,
        "sha256": "3f7afe231a0481df020eca0cc7287d07aceb7a9f8c4b85d474239059aacd78b8",
    }
    assert summary["project_counts"] == {"Project03": 10370, "Project04": 2075}
    assert summary["project04_evaluation_summaries"]["row_count"] == 229
    assert not (RESULTS / "source_inventory.csv").exists()


def test_conflict_table_has_required_schema() -> None:
    conflicts = pd.read_csv(RESULTS / "conflicts.csv")
    assert list(conflicts.columns) == [
        "experiment", "seed", "target", "field", "source_a", "value_a",
        "source_b", "value_b", "recommended_next_check",
    ]
    assert not conflicts.empty


def test_results_readme_navigates_public_result_layers() -> None:
    text = (RESULTS / "README.md").read_text(encoding="utf-8")
    for name in (
        "stage1/", "stage2/", "portfolio/", "metadata/", "provenance/",
        "experiment_registry.csv", "metric_definitions.json", "conflicts.csv",
    ):
        assert name in text

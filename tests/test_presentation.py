"""Consistency checks for bilingual public presentation."""

from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
DOCS = ROOT / "docs"
RESULTS_README = ROOT / "results" / "README.md"
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

EXPECTED_TARGET_FORMULA = r"""$$
y_{i,t}=
\frac{\mathrm{AdjOpen}_{i,t+6}/\mathrm{AdjOpen}_{i,t+1}}
{\mathrm{Open}_{\mathrm{CSI500},t+6}/\mathrm{Open}_{\mathrm{CSI500},t+1}}
-1.
$$"""

STAGE1_NAME = "master_d512_mse"
STAGE2_HEAD_EXECUTIONS = (
    "20260826_exp102_headscreen_m63_d512_t8_s2_seed42",
    "20260826_exp104_headscreen_m63_d512_t4_s4_seed42",
)
LATE_STAGE2 = ROOT / "results" / "stage2" / "late_regime_results.csv"


def _public_markdown() -> str:
    paths = [README, README_ZH, RESULTS_README] + sorted(DOCS.rglob("*.md"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _selected_prediction_rows():
    stage1 = pd.read_csv(ROOT / "results/stage1/main_results.csv")
    stage1_row = stage1.loc[
        stage1["public_display_name"].eq(STAGE1_NAME)
        & stage1["target"].eq("Overall")
    ].iloc[0]
    stage2 = pd.read_csv(ROOT / "results/stage2/head_sensitivity.csv")
    stage2_rows = stage2.set_index("historical_execution_id").loc[
        list(STAGE2_HEAD_EXECUTIONS)
    ]
    return stage1_row, stage2_rows


def _parameterized_setting(stage: str, row, *, ensemble: bool = False) -> str:
    market = str(row["market_state"]).replace("Market", "M")
    seed = "Seeds42/43/44 Ensemble" if ensemble else f"Seed{int(row['seed'])}"
    return (
        f"{stage} · {market} · D{int(row['d_model'])} · "
        f"H{int(row['temporal_heads'])}×{int(row['spatial_heads'])} · {seed}"
    )


def test_readme_prediction_table_has_explicit_seed_and_aggregation_semantics():
    stage1, stage2 = _selected_prediction_rows()
    assert stage1["aggregation"] == "raw_prediction_ensemble"
    assert stage1["seed"] == "ensemble"
    assert bool(stage1["historical_raw_prediction_ensemble"])
    assert (stage1["temporal_heads"], stage1["spatial_heads"]) == (4, 2)
    assert stage2["seed"].eq(42).all()
    assert stage2["target"].eq("ALL").all()
    assert not stage2["historical_raw_prediction_ensemble"].astype(bool).any()

    english = README.read_text(encoding="utf-8")
    chinese = README_ZH.read_text(encoding="utf-8")
    assert "mean ± sample standard deviation across three independently trained seeds" not in english
    assert "Seed / aggregation" not in english
    assert "## Selected Results" in english
    assert "## 部分实验结果" in chinese
    assert _parameterized_setting("Stage 1", stage1, ensemble=True) in english
    assert _parameterized_setting("Stage 1", stage1, ensemble=True) in chinese
    for _, row in stage2.iterrows():
        setting = _parameterized_setting("Stage 2", row)
        assert setting in english
        assert setting in chinese
    for execution_id in STAGE2_HEAD_EXECUTIONS:
        assert execution_id not in english
        assert execution_id not in chinese


def test_prediction_table_values_and_bp_to_permille_conversion_are_canonical():
    stage1, stage2 = _selected_prediction_rows()
    texts = (README.read_text(encoding="utf-8"), README_ZH.read_text(encoding="utf-8"))

    stage1_metrics = "| {:.4f} | {:.4f} | {:.3f} | {:.3f} |".format(
        stage1["IC"], stage1["RankIC"],
        stage1["Top5_bp"] / 10.0, stage1["Top10_bp"] / 10.0,
    )
    assert all(stage1_metrics in text for text in texts)

    for _, row in stage2.iterrows():
        metrics = (
            "| {ic:.4f} | {ric:.4f} | {top5:.3f} | {top10:.3f} |"
        ).format(
            ic=row["IC"], ric=row["RankIC"],
            top5=row["Top5_bp"] / 10.0,
            top10=row["Top10_bp"] / 10.0,
        )
        assert all(metrics in text for text in texts)

    assert "shown in ‰" in texts[0]
    assert "以 ‰ 展示" in texts[1]
    assert all("1 bp = 0.1‰" not in text for text in texts)


def test_later_period_table_preserves_historical_rows_and_stage1_csv_evidence():
    texts = (README.read_text(encoding="utf-8"), README_ZH.read_text(encoding="utf-8"))

    stage1 = pd.read_csv(ROOT / "results/stage1/main_results.csv")
    stage1 = stage1.loc[stage1["public_display_name"].eq(STAGE1_NAME)].set_index("target")
    for target in ("2025H1", "2025H2", "2026H1_partial"):
        row = stage1.loc[target]
        rendered = "{:.3f}‰ / {:.3f}‰".format(
            row["Top5_bp"] / 10.0,
            row["Top10_bp"] / 10.0,
        )
        assert all(rendered in text for text in texts)

    late = pd.read_csv(LATE_STAGE2)
    for _, group in late.groupby("experiment", sort=False):
        by_target = group.set_index("target")
        row = group.iloc[0]
        label = _parameterized_setting("Stage 2", row)
        values = (
            by_target.loc["2025H2", "Top5_bp"] / 10.0,
            by_target.loc["2025H2", "Top10_bp"] / 10.0,
            by_target.loc["2026H1_partial", "Top5_bp"] / 10.0,
            by_target.loc["2026H1_partial", "Top10_bp"] / 10.0,
        )
        rendered_values = " | ".join("{:.3f}‰".format(value) for value in values)
        rendered_row = f"| {label} | {rendered_values} |"
        assert all(rendered_row in text for text in texts)


def test_prediction_top_fractions_are_not_portfolio_long_semantics():
    for text in (README.read_text(encoding="utf-8"), README_ZH.read_text(encoding="utf-8")):
        assert "Top 5%" in text
        assert "Top 10%" in text
        assert re.search(r"(?<![A-Za-z])Top5(?![_A-Za-z])", text) is None
        assert re.search(r"(?<![A-Za-z])Top10(?![_A-Za-z])", text) is None
    assert "long portfolio" in README.read_text(encoding="utf-8").lower()
    assert "多头组合" in README_ZH.read_text(encoding="utf-8")


def test_stage2_ensemble_and_partial_coverage_are_not_misrepresented():
    for path in sorted((ROOT / "results/stage2").glob("*.csv")):
        frame = pd.read_csv(path)
        if "historical_raw_prediction_ensemble" in frame:
            assert not frame["historical_raw_prediction_ensemble"].fillna(False).astype(bool).any()
        if "matched_control" in frame:
            assert not frame["matched_control"].fillna(False).astype(bool).any()

    robustness = pd.read_csv(ROOT / "results/stage2/seed_robustness.csv")
    d128 = robustness.loc[
        robustness["public_display_name"].eq(
            "capacity_d128_market63_t4_s2_seed_robustness"
        )
    ].iloc[0]
    assert d128["completion_status"] == "partial_seed_coverage"
    text = README.read_text(encoding="utf-8")
    assert "Stage 2 D128 has partial seed coverage" not in text
    assert "no same-protocol Stage 2 Alpha158-only control is available" in text


def test_selected_portfolio_uses_verified_parameterized_setting_names():
    expected = (
        "LightGBM · Stage 1 · M63 · Seed42",
        "Stage 1 · M63 · D128 · H4×1 · Seeds42/43/44 Ensemble",
        "Stage 2 · M63 · D256 · H4×2 · Seed43",
        "Stage 2 · M63 · D512 · H4×2 · Seed42",
    )
    for text in (
        README.read_text(encoding="utf-8"),
        README_ZH.read_text(encoding="utf-8"),
    ):
        for setting in expected:
            assert setting in text


def test_readme_positions_the_project_for_ml_and_quant_reviewers():
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()
    required = (
        "daily and minute-level a-share market data",
        "five-trading-day excess return relative to the csi500",
        "complete rolling out-of-sample research pipeline",
        "dynamic cross-section of a-shares",
        "8-trading-day sequence",
        "158 daily alpha158 features",
        "248 engineered intraday price-volume features",
        "market conditions are represented separately",
        "one prediction score for each stock",
        "rankic",
    )
    for phrase in required:
        assert phrase in lowered
    for pattern in (
        r"\bllm\b", r"\bfoundation model\b", r"\bgenerative ai\b",
        r"\bmultimodal ai\b", r"\bsota\b", r"\bproduction-grade\b",
    ):
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_readme_stage2_tensor_description_matches_both_market_layouts():
    from market_guided_stock_alpha.dataset.layout import FeatureLayout

    market63 = FeatureLayout.stage2("market63")
    market84 = FeatureLayout.stage2("market84")
    assert market63.lookback == market84.lookback == 8
    assert market63.stock_dim == market84.stock_dim == 406
    assert (market63.market_dim, market84.market_dim) == (63, 84)

    text = README.read_text(encoding="utf-8")
    assert r"N \times 8 \times 406" in text
    assert r"8 \times D_m" in text
    assert r"D_m \in \{63,84\}" in text
    assert "broadcast across the current cross-section only inside the model forward pass" in text


def test_readmes_present_intraday_design_findings_and_open_directions():
    english = README.read_text(encoding="utf-8")
    chinese = README_ZH.read_text(encoding="utf-8")

    for text in (english, chinese):
        assert "docs/intraday_features.md" in text
        for term in ("F001–F248", "248D", "Spectral", "Geometry", "Multiscale"):
            assert term in text
        assert "current model directly consumes raw minute sequences" not in text.lower()

    assert "## Experimental Findings" in english
    assert "## 实验观察与发现" in chinese
    assert "## Open Directions" in english
    assert "## 后续研究方向" in chinese
    assert "minute data → stock-day 248D representation → 8-day sequence → Transformer" in english
    assert "分钟行情 → stock-day 248D 表示 → 8 日序列 → Transformer" in chinese


def test_readme_data_dates_frequencies_and_feature_origins_are_supported():
    english = README.read_text(encoding="utf-8")
    chinese = README_ZH.read_text(encoding="utf-8")
    for text in (english, chinese):
        assert "2016" in text
        assert "2026-04-27" in text
        assert "2023" in text
        assert "2026H1" in text
        assert "semiannual" in text.lower() or "半年" in text
        assert "quarterly" in text.lower() or "季度" in text
        assert "monthly" in text.lower() or "月度" in text
        assert "Alpha158" in text
        assert "Qlib" in text
        assert "248" in text

    assert "project-designed Intraday Price-Volume Features" in english
    assert "本项目设计" in chinese
    assert "proprietary alpha data" not in english.lower()
    assert "company factor library" in english.lower()
    assert "did not use a company factor library" in english


def test_readme_opening_and_main_copy_avoid_defensive_audit_language():
    english = README.read_text(encoding="utf-8")
    chinese = README_ZH.read_text(encoding="utf-8")
    assert "not the stock universe" not in english
    assert "CSI500 不是股票池" not in chinese
    assert "ddof=1" not in english
    assert "ddof=1" not in chinese
    assert "deep-learning research and engineering framework" not in english
    assert "investigates whether project-designed minute-level price-volume features" not in english
    assert "本项目使用 **A 股日频与分钟级行情数据**" in chinese
    assert "股票数量动态变化的 A 股横截面" in chinese


def test_alpha158_attribution_and_no_paper_or_citation_section():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Microsoft Qlib Alpha158" in notices
    assert "Alpha158DL" in notices
    assert "Copyright (c) Microsoft Corporation." in notices
    for text in (README.read_text(encoding="utf-8"), README_ZH.read_text(encoding="utf-8")):
        assert re.search(r"^## (?:Paper|Citation|Cite this work)$", text, re.MULTILINE) is None


def test_canonical_model_docs_do_not_use_migration_era_positioning():
    source = ROOT / "src/market_guided_stock_alpha/modeling"
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(source.rglob("*.py")))
    assert "03/04 research code" not in text
    assert "completed 04 branch" not in text


def test_public_docs_do_not_expose_internal_or_migration_paths():
    text = _public_markdown()
    for pattern in (
        r"/mnt/data01", r"[A-Za-z]:\\Users\\", r"03_master_replication",
        r"(?:^|[\s`/])04/", r"(?:\d{1,3}\.){3}\d{1,3}",
    ):
        assert re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is None


def test_bilingual_language_links_mermaid_latex_and_controls_are_valid():
    english = README.read_text(encoding="utf-8")
    chinese = README_ZH.read_text(encoding="utf-8")
    assert english.startswith("# End-to-End Stock Transformer\n")
    assert chinese.startswith("# End-to-End Stock Transformer\n")
    assert "**English | [简体中文](README.zh-CN.md)**" in english
    assert "**[English](README.md) | 简体中文**" in chinese
    for path, text in ((README, english), (README_ZH, chinese)):
        assert text.count("```mermaid") == 1
        assert "flowchart TD" in text
        assert "A --> B --> C --> D --> E --> F --> G" in text
        assert EXPECTED_TARGET_FORMULA in text
        unexpected = [
            (offset, ord(char)) for offset, char in enumerate(text)
            if unicodedata.category(char) == "Cc" and char not in "\n\r\t"
        ]
        assert unexpected == [], (path, unexpected)

    assert "Daily and minute stock/index OHLCV and amount" not in english
    assert "Intraday Market Features (192D, exploratory)" not in english
    assert "股票与指数日频/分钟 OHLCV 和 amount" not in chinese


def test_public_markdown_math_avoids_unsupported_github_macros():
    paths = (
        [README, README_ZH, RESULTS_README, ROOT / "THIRD_PARTY_NOTICES.md"]
        + sorted(DOCS.rglob("*.md"))
    )
    for path in paths:
        assert r"\operatorname{" not in path.read_text(encoding="utf-8")

    intraday_features = (DOCS / "intraday_features.md").read_text(encoding="utf-8")
    for legacy_delimiter in (r"\(", r"\)", r"\[", r"\]"):
        assert legacy_delimiter not in intraday_features
    assert r"\#" not in intraday_features
    assert "\n=\n" not in intraday_features

    for path in (README, README_ZH):
        assert "\n=\n" not in path.read_text(encoding="utf-8")


def test_public_markdown_has_no_image_links_or_broken_local_links():
    markdown = [README, README_ZH, RESULTS_README] + sorted(DOCS.rglob("*.md"))
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        assert image_pattern.search(text) is None, path
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[a-z][a-z0-9+.-]*://", target, re.I):
                continue
            assert (path.parent / target).resolve().exists(), (path, target)

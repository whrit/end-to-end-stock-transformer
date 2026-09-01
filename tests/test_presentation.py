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

STAGE1_ID = (
    "master_warm_semiannual_d128_b10_h4x1_lr1e-05_p0.5_"
    "abl-none_xc9dc8fcb65|loss=MSE"
)
STAGE2_IDS = (
    "capacity_d256_market63_t4_s2_seed_robustness",
    "capacity_d512_market63_t4_s2_seed_robustness",
    "market84_d256_market84_t4_s2_seed_robustness",
)


def _public_markdown() -> str:
    paths = [README, README_ZH, RESULTS_README] + sorted(DOCS.rglob("*.md"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _selected_prediction_rows():
    stage1 = pd.read_csv(ROOT / "results/stage1/seed_robustness.csv")
    stage1_row = stage1.loc[stage1["public_display_name"].eq(STAGE1_ID)].iloc[0]
    stage2 = pd.read_csv(ROOT / "results/stage2/seed_robustness.csv")
    stage2_rows = stage2.set_index("public_display_name").loc[list(STAGE2_IDS)]
    return stage1_row, stage2_rows


def test_readme_prediction_table_uses_only_complete_three_seed_statistics():
    stage1, stage2 = _selected_prediction_rows()
    assert stage1["completion_status"] == "complete_3seed"
    assert stage1["aggregation"] == "seed_robustness_mean_std"
    assert stage1["n_seeds"] == 3
    assert stage2["completion_status"].eq("complete_3seed").all()
    assert stage2["aggregation"].eq("seed_robustness_mean_std").all()

    text = README.read_text(encoding="utf-8")
    assert "mean ± sample standard deviation across three independently trained seeds" in text
    assert "Stage 1 · Alpha158 · Market63 · D128" in text
    assert "Stage 2 · Alpha158 + intraday · Market63 · D256" in text
    assert "Stage 2 · Alpha158 + intraday · Market63 · D512" in text
    assert "Stage 2 · Alpha158 + intraday · Market84 · D256" in text

    # Historical raw-score ensemble single points remain documented but are
    # not rows in the cross-stage prediction table.
    assert "| MASTER D512 | 3-seed raw-score ensemble |" not in text
    assert "Stage 1 prediction-level ensemble results" in text


def test_prediction_table_values_and_bp_to_permille_conversion_are_canonical():
    stage1, stage2 = _selected_prediction_rows()
    texts = (README.read_text(encoding="utf-8"), README_ZH.read_text(encoding="utf-8"))

    stage1_metrics = (
        "{:.4f} ± {:.4f} | {:.4f} ± {:.4f} | — | — |".format(
            stage1["IC_mean"], stage1["IC_std"],
            stage1["RankIC_mean"], stage1["RankIC_std"],
        )
    )
    assert all(stage1_metrics in text for text in texts)

    for _, row in stage2.iterrows():
        # Canonical sources remain in bp; public README presentation is per mille.
        metrics = (
            "{ic:.4f} ± {ic_std:.4f} | {ric:.4f} ± {ric_std:.4f} | "
            "{top5:.3f} ± {top5_std:.3f} | {top10:.3f} ± {top10_std:.3f} |"
        ).format(
            ic=row["IC_mean"], ic_std=row["IC_std"],
            ric=row["RankIC_mean"], ric_std=row["RankIC_std"],
            top5=row["Top5_bp_mean"] / 10.0,
            top5_std=row["Top5_bp_std"] / 10.0,
            top10=row["Top10_bp_mean"] / 10.0,
            top10_std=row["Top10_bp_std"] / 10.0,
        )
        assert all(metrics in text for text in texts)

    assert "shown in ‰" in texts[0]
    assert "单位为 ‰" in texts[1]
    assert all("1 bp = 0.1‰" not in text for text in texts)


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
    assert "Stage 2 D128 has partial seed coverage" in text
    assert "no same-protocol Stage 2 Alpha158-only control is available" in text


def test_readme_positions_the_project_for_ml_and_quant_reviewers():
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()
    required = (
        "deep-learning research and engineering framework",
        "multivariate sequential prediction",
        "cross-sectional stock-return forecasting",
        "market-guided transformers",
        "intraday price-volume representations",
        "dynamic all a-shares",
        "csi500",
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
    assert "investigates whether project-designed minute-level price-volume features" in english
    assert "重点研究自设计的分钟级量价特征能否" in chinese


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

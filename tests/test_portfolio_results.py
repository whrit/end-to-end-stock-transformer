"""Public portfolio aggregates and private-source boundary checks."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results/portfolio/portfolio_results.csv"
PRIVATE_SOURCE = ROOT / "backtesting"
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "docs/portfolio_evaluation.md",
    ROOT / "docs/results_provenance.md",
)

EXPECTED_COLUMNS = [
    "public_display_name", "historical_name", "model", "feature_family",
    "market_state", "d_model", "seed_or_ensemble", "date_start", "date_end",
    "comparison_status", "long_total_return", "long_annual_return",
    "long_sharpe", "long_short_total_return", "long_short_annual_return",
    "long_short_sharpe", "ic_mean", "icir", "turnover",
    "source_descriptor", "notes",
]

SOURCE_TO_PUBLIC = {
    "TopRetTol": "long_total_return",
    "TopRetYL": "long_annual_return",
    "TopSharpYL": "long_sharpe",
    "LSRetTol": "long_short_total_return",
    "LSRetYL": "long_short_annual_return",
    "LSSharpYL": "long_short_sharpe",
    "ICMean_estu.r": "ic_mean",
    "ICIR_estu.r": "icir",
    "turnover_ratemean": "turnover",
}


def _read_docx_key_values(path: Path):
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    values = {}
    for row in root.findall(".//w:tr", namespace):
        cells = []
        for cell in row.findall("w:tc", namespace):
            text = "".join(node.text or "" for node in cell.findall(".//w:t", namespace)).strip()
            cells.append(text)
        if len(cells) >= 2 and cells[0]:
            values[cells[0]] = cells[1]
    return values


def test_portfolio_csv_schema_and_public_semantics():
    frame = pd.read_csv(CSV_PATH)
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 10
    assert frame["public_display_name"].is_unique
    assert frame["date_end"].eq("2026-04-27").all()
    assert set(frame["date_start"]) == {"2023-01-01", "2023-01-03"}
    assert frame["source_descriptor"].eq(
        "sanitized aggregate statistics from historical institutional backtest export"
    ).all()

    joined_headers = " ".join(frame.columns)
    for internal_field in (
        "TopOOSRetYL", "TopOOSSharpYL", "LSOOSRetYL", "LSOOSSharpYL"
    ):
        assert internal_field not in joined_headers
    assert all(not column.lower().startswith("top") for column in frame.columns)


def test_public_portfolio_values_match_private_docx_exports_when_available():
    # The source directory is intentionally absent from a public clone. When it
    # exists locally, verify every published aggregate directly against DOCX XML.
    if not PRIVATE_SOURCE.exists():
        return

    frame = pd.read_csv(CSV_PATH).set_index("historical_name")
    docx_files = sorted(PRIVATE_SOURCE.rglob("*.docx"))
    assert len(docx_files) == len(frame) == 10
    recovered_names = set()
    for path in docx_files:
        source = _read_docx_key_values(path)
        historical_name = source["Factor"]
        assert historical_name in frame.index
        recovered_names.add(historical_name)
        public = frame.loc[historical_name]
        for source_name, public_name in SOURCE_TO_PUBLIC.items():
            assert float(public[public_name]) == float(source[source_name])
        start, end = source["Dates"].split("~")
        assert public["date_start"] == pd.Timestamp(start).strftime("%Y-%m-%d")
        assert public["date_end"] == pd.Timestamp(end).strftime("%Y-%m-%d")
    assert recovered_names == set(frame.index)


def test_private_backtesting_is_ignored_and_not_copied_publicly():
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/backtesting/" in ignore_lines
    for path in ROOT.rglob("*"):
        if not path.is_file() or PRIVATE_SOURCE in path.parents:
            continue
        assert path.suffix.lower() not in {".docx", ".xlsx"}, path


def test_public_portfolio_material_has_no_internal_fields_paths_or_topret_name():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
    text += "\n" + CSV_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "TopOOSRetYL", "TopOOSSharpYL", "LSOOSRetYL", "LSOOSSharpYL",
        "TopRet", "/mnt/data01",
    ):
        assert forbidden not in text
    assert re.search(r"[A-Za-z]:\\Users\\", text) is None
    assert re.search(r"(?:\d{1,3}\.){3}\d{1,3}", text) is None


def test_selected_portfolio_rows_are_bilingual_and_match_public_csv():
    frame = pd.read_csv(CSV_PATH).set_index("public_display_name")
    selected = (
        "lightgbm_alpha158_market63",
        "master_alpha158_d128_market63_3seed_ensemble",
        "intraday_price_volume_master_d256_market63_seed43",
        "intraday_price_volume_master_d512_market63_seed42",
    )
    texts = (
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "README.zh-CN.md").read_text(encoding="utf-8"),
    )
    for setting in selected:
        row = frame.loc[setting]
        metrics = "| {:.1f}% | {:.3f} | {:.1f}% | {:.3f} | {:.1f}% |".format(
            row["long_annual_return"] * 100.0, row["long_sharpe"],
            row["long_short_annual_return"] * 100.0,
            row["long_short_sharpe"], row["turnover"] * 100.0,
        )
        assert all(metrics in text for text in texts)
    assert "| Signal | IC |" not in texts[0]
    assert "| 信号 | IC |" not in texts[1]
    assert "Mean Turnover" in texts[0]
    assert "平均换手率" in texts[1]


def test_portfolio_public_docs_disclose_method_limits_without_overclaiming():
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    detail = (ROOT / "docs/portfolio_evaluation.md").read_text(encoding="utf-8")
    for text in (english, detail):
        assert "transaction fees were not deducted" in text.lower()
        assert "not a controlled comparison" in text.lower()
        assert "long portfolio" in text.lower()
    assert "## Supplementary Portfolio Evaluation" in english
    assert "not used as the primary measure of predictive model quality" in english.lower()
    assert "higher portfolio return and Sharpe, together with higher mean turnover" not in english

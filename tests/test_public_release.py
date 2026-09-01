"""Pre-Git public release boundary checks."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_ROOTS = (
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "docs",
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "configs",
    ROOT / "results",
    ROOT / "pyproject.toml",
)
TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".toml", ".yaml", ".yml", ".txt", ".sh"}


def _text_files():
    for root in PUBLIC_TEXT_ROOTS:
        if root.is_file():
            yield root
        else:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                    yield path


def test_migration_and_one_time_recovery_code_are_absent():
    assert not (ROOT / "03_master_replication").exists()
    assert not (ROOT / "04").exists()
    assert not (ROOT / "scripts" / "build_recovered_results.py").exists()
    assert not (ROOT / "scripts" / "generate_figures.py").exists()
    assert not (ROOT / "results" / "source_inventory.csv").exists()
    assert not (ROOT / "docs" / "figures").exists()


def test_license_and_upstream_notice_are_complete():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    assert "End-to-End Stock Transformer contributors" in license_text
    assert "Copyright (c) 2025 Data Management Technology and AI" in notice
    assert "Exact historical upstream commit" in notice
    assert "src/market_guided_stock_alpha/modeling/master.py" in notice


def test_public_text_has_no_internal_paths_or_credentials():
    strong_patterns = (
        re.compile(r"/mnt/data01", re.I),
        re.compile(r"intern001", re.I),
        re.compile(r"192\.168\.\d{1,3}\.\d{1,3}"),
        re.compile(r"(?<!\d)10\.(?:\d{1,3}\.){2}\d{1,3}(?!\d)"),
        re.compile(r"C:\\Users\\", re.I),
        re.compile(r"D:\\swhy_intern", re.I),
        re.compile(r"DataYes|通联", re.I),
        re.compile(r"ssh://", re.I),
        re.compile(
            r"(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]+['\"]",
            re.I,
        ),
    )
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in strong_patterns:
            assert pattern.search(text) is None, (path, pattern.pattern)


def test_gitignore_excludes_local_material_without_hiding_results():
    lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required = {
        "*.tar.gz",
        "project03_github_handoff_*/",
        "project04_results_export_*/",
        "_result_recovery/",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "outputs/",
        "cache/",
        "logs/",
    }
    assert required.issubset(lines)
    assert "results/" not in lines
    assert "*.csv" not in lines
    assert "*.json" not in lines


def test_pyproject_declares_public_readme_license_and_python38_floor():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "end-to-end-stock-transformer"' in text
    assert 'description = "End-to-end Transformer research for A-share cross-sectional return forecasting"' in text
    assert 'version = "0.1.0"' in text
    assert 'readme = "README.md"' in text
    assert 'requires-python = ">=3.8"' in text
    assert 'license = {file = "LICENSE"}' in text
    assert "matplotlib" not in text
    assert "presentation" not in text
    assert (ROOT / "src/market_guided_stock_alpha").is_dir()

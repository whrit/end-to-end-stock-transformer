"""Registry consistency checks for the public Intraday 248D specification."""

from pathlib import Path
import re
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.features.intraday.registry import FACTOR_REGISTRY


DOCUMENT = ROOT / "docs" / "intraday_features.md"
MAPPING_ROW = re.compile(
    r"^\| (F\d{3}) \| `([^`]+)` \| `([^`]+)` \| "
    r"([^|]+?) \| `([^`]+)` \|$",
    re.MULTILINE,
)


def test_intraday_documentation_covers_the_frozen_registry_exactly_once():
    text = DOCUMENT.read_text(encoding="utf-8")
    rows = MAPPING_ROW.findall(text)

    assert not any(
        unicodedata.category(character) == "Cc"
        and character not in "\n\r"
        for character in text
    )
    assert len(rows) == 248
    documented = [(factor_id, name, category) for factor_id, name, category, _, _ in rows]
    assert documented == list(FACTOR_REGISTRY)
    assert len({row[0] for row in rows}) == 248
    assert len({row[1] for row in rows}) == 248

    feature_root = SOURCE_ROOT / "market_guided_stock_alpha" / "features"
    for _factor_id, _name, _category, definition_section, implementation in rows:
        assert "### {}".format(definition_section.strip()) in text
        assert (feature_root / implementation).is_file()

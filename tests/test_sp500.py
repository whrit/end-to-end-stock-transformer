"""Point-in-time S&P 500 universe: identity through renames and ticker reuse,
FIGI verification, membership, and delisting classification (synthetic)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from market_guided_stock_alpha.data.contracts import DataQualityError, normalize_security_code
from market_guided_stock_alpha.us_data.sp500 import build_sp500_universe, parse_history

DAYS = pd.bdate_range("2024-01-01", periods=30)


def prices(name, first, last):
    """Bars for one company on DAYS[first..last]; BUST collapses at the end."""
    span = DAYS[first:last + 1]
    base = {"A": 100.0, "C1": 300.0, "GONE": 50.0, "REUSE": 10.0, "BIG": 80.0, "BUST": 40.0, "ACQ": 60.0}[name]
    close = np.full(len(span), base) + np.arange(first, last + 1) * 0.01
    if name == "BUST":
        close[-5:] = [30.0, 20.0, 12.0, 8.0, 5.0]
    return pd.DataFrame({"trade_date": span, "open": close, "high": close, "low": close, "close": close,
                         "volume": 1e6, "vwap": close})


SPANS = {"A": (0, 29), "C1": (0, 29), "GONE": (0, 19), "REUSE": (25, 29), "BIG": (0, 29), "BUST": (0, 24),
         "ACQ": (0, 20)}


def resolve(symbol, asof):
    """(company, first, last) pieces a ticker returns: ``asof`` a date follows
    the company through renames; None = every bar printed under the symbol."""
    whole = lambda company: [(company,) + SPANS[company]]
    if symbol == "OLD":
        # As of its old name the company reads a shorter history (like KDP for
        # Dr Pepper Snapple): same prices on shared days, different coverage.
        return [("C1", 3, 29)] if asof is not None else [("C1", 0, 9)]
    if symbol == "NEW":
        return whole("C1") if asof is not None else [("C1", 10, 29)]
    if symbol == "GONE":
        return whole("GONE") + ([] if asof is not None else whole("REUSE"))
    return whole({"AAA": "A", "BIG": "BIG", "BUST": "BUST", "ACQ": "ACQ"}[symbol])


def fetch_bars(symbols, asof, adjustment):
    parts = [
        prices(company, first, last).assign(alpaca_symbol=symbol)
        for symbol in symbols
        for company, first, last in resolve(symbol, asof)
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["alpaca_symbol", "trade_date"])


LISTED = {"AAA": ("BBG000000AAA", "Common Stock"), "NEW": ("BBG0000000C1", "Common Stock"),
          "BIG": ("BBG000000BIG", "REIT"), "GONE": ("BBG00000REUS", "Common Stock")}
UNLISTED = {"ACQ": ("BBG000000ACQ", "Common Stock"), "GONE": ("BBG0000GONE1", "Common Stock")}


def map_figi(tickers, include_unlisted):
    rows = []
    for ticker in tickers:
        figi, kind = (UNLISTED if include_unlisted else LISTED).get(ticker, (None, None))
        rows.append({"ticker": ticker, "share_class_figi": figi, "security_type2": kind,
                     "market_sector": "Equity", "name": ticker, "mapping_note": "mapped" if figi else "none"})
    return pd.DataFrame(rows)


HISTORY = parse_history(
    "date,tickers\n"
    '2023-12-01,"AAA,OLD,GONE,BUST,ACQ"\n'
    '{},"AAA,NEW,GONE,BUST,ACQ"\n'
    '{},"AAA,NEW,BUST,BIG,ACQ"\n'
    '{},"AAA,NEW,BUST,BIG"\n'
    '{},"AAA,NEW,BIG"\n'.format(DAYS[10].date(), DAYS[15].date(), DAYS[21].date(), DAYS[27].date())
)

ASSETS = pd.DataFrame({"alpaca_symbol": ["BUST"], "status": ["inactive"], "exchange": ["NYSE"]})


class Sp500UniverseTest(unittest.TestCase):
    def setUp(self):
        self.result = build_sp500_universe(HISTORY, DAYS, fetch_bars, map_figi, ASSETS)
        self.securities = self.result["securities"].set_index("sec_code")

    def test_rename_is_one_company_with_continuous_membership(self):
        row = self.securities.loc["BBG0000000C1"]
        self.assertEqual(row["sp500_tickers"], "OLD|NEW")
        self.assertEqual(row["identity"], "figi_current")
        members = self.result["membership"]
        self.assertEqual(int((members["sec_code"] == "BBG0000000C1").sum()), 30)

    def test_reused_ticker_never_lends_its_figi(self):
        code = "GONE.US.{:%Y%m%d}".format(DAYS[19])
        self.assertEqual(self.securities.loc[code, "identity"], "fallback_ticker_reused")
        self.assertNotIn("BBG00000REUS", self.securities.index)
        members = self.result["membership"]
        self.assertEqual(int((members["sec_code"] == code).sum()), 15)  # member days 0..14 only

    def test_reit_kept_and_unmapped_bankruptcy_named_by_last_bar(self):
        self.assertEqual(self.securities.loc["BBG000000BIG", "security_type2"], "REIT")
        self.assertEqual(self.securities.loc["BUST.US.{:%Y%m%d}".format(DAYS[24]), "identity"], "fallback_unmapped")

    def test_acquired_company_with_unused_ticker_keeps_its_delisted_figi(self):
        self.assertEqual(self.securities.loc["BBG000000ACQ", "identity"], "figi_delisted")

    def test_delisting_kinds_and_exit_factors(self):
        delisted = self.result["delistings"].set_index("sec_code")
        gone = delisted.loc["GONE.US.{:%Y%m%d}".format(DAYS[19])]
        bust = delisted.loc["BUST.US.{:%Y%m%d}".format(DAYS[24])]
        self.assertEqual((gone["kind"], gone["exit_factor"]), ("merger", 1.0))
        self.assertEqual(delisted.loc["BBG000000ACQ", "kind"], "merger")
        self.assertEqual((bust["kind"], bust["exchange"]), ("performance", "NYSE"))
        self.assertAlmostEqual(bust["exit_factor"], 0.70)
        self.assertEqual(len(delisted), 3)

    def test_counts(self):
        counts = self.result["counts"]
        self.assertEqual((counts["episodes"], counts["companies"]), (7, 6))
        self.assertEqual(counts["companies_with_several_tickers"], 1)

    def test_fallback_code_contract(self):
        self.assertEqual(normalize_security_code("sivb.us.20230309"), "SIVB.US.20230309")
        self.assertEqual(normalize_security_code("BRK.B.US.20230309"), "BRK.B.US.20230309")
        with self.assertRaises(DataQualityError):
            normalize_security_code("SIVB.US.2023")


if __name__ == "__main__":
    unittest.main()

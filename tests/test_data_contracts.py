"""Focused tests for canonical public market-data contracts."""

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.data import (
    DataQualityError,
    SchemaError,
    TradingCalendar,
    validate_daily_index,
    validate_daily_stock,
    validate_membership,
)


def daily_stock():
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "sec_code": ["000001.sz", "600000.SH"],
            "open": [10.0, 20.0],
            "high": [10.5, 20.5],
            "low": [9.5, 19.5],
            "close": [10.2, 20.2],
            "vwap": [10.1, 20.1],
            "volume": [1000, 2000],
        }
    )


def daily_index():
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "index_code": ["000300.SH", "000905.SH"],
            "open": [3500.0, 5000.0],
            "close": [3510.0, 5010.0],
            "amount": [1e11, 8e10],
        }
    )


class DataContractTest(unittest.TestCase):
    def test_valid_daily_stock_and_index_are_normalized(self):
        stock = validate_daily_stock(daily_stock())
        index = validate_daily_index(daily_index())
        self.assertEqual(stock["sec_code"].tolist(), ["000001.SZ", "600000.SH"])
        self.assertTrue(pd.api.types.is_datetime64_ns_dtype(stock["trade_date"]))
        self.assertEqual(index["index_code"].tolist(), ["000300.SH", "000905.SH"])

    def test_duplicate_keys_are_rejected(self):
        duplicate = pd.concat([daily_stock().iloc[[0]], daily_stock().iloc[[0]]])
        with self.assertRaisesRegex(DataQualityError, "duplicate"):
            validate_daily_stock(duplicate)

    def test_invalid_dates_codes_and_missing_columns_are_rejected(self):
        invalid_date = daily_stock()
        invalid_date.loc[0, "trade_date"] = pd.NaT
        with self.assertRaisesRegex(DataQualityError, "invalid dates"):
            validate_daily_stock(invalid_date)

        invalid_code = daily_stock()
        invalid_code.loc[0, "sec_code"] = "000001"
        with self.assertRaisesRegex(DataQualityError, "exchange guessing is disabled"):
            validate_daily_stock(invalid_code)

        with self.assertRaisesRegex(SchemaError, "missing required columns.*vwap"):
            validate_daily_stock(daily_stock().drop(columns="vwap"))

    def test_membership_and_calendar_are_thin_canonical_contracts(self):
        membership = validate_membership(
            pd.DataFrame(
                {
                    "trade_date": ["2024-01-02"],
                    "sec_code": ["000001.SZ"],
                    "index_code": ["000905.SH"],
                    "weight": [0.01],
                }
            )
        )
        self.assertEqual(list(membership.columns), [
            "trade_date", "sec_code", "index_code", "weight"
        ])
        calendar = TradingCalendar(
            pd.DataFrame(
                {
                    "calendar_date": pd.date_range("2024-01-01", periods=5),
                    "is_open": [0, 1, 1, 1, 1],
                }
            )
        )
        self.assertEqual(
            calendar.next_trading_date("2024-01-02", 2),
            pd.Timestamp("2024-01-04"),
        )
        self.assertEqual(
            calendar.previous_trading_date("2024-01-05", 2),
            pd.Timestamp("2024-01-03"),
        )


if __name__ == "__main__":
    unittest.main()

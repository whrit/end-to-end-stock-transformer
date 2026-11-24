import unittest

import numpy as np
import pandas as pd

from market_guided_stock_alpha.us_data.crsp import COLUMNS, ETF_PERMNOS, build_crsp_tables

DAYS = pd.bdate_range("2020-01-02", periods=12)


def rows_for(permno, days, price, cap, exchange="N", returns=None, kind=None):
    frame = pd.DataFrame(
        {
            "PERMNO": permno, "DlyCalDt": days, "Ticker": "T{}".format(permno), "IssuerNm": "N{}".format(permno),
            "PrimaryExch": exchange, "SecurityType": "EQTY", "SecuritySubType": "COM", "ShareType": "NS",
            "USIncFlg": "Y", "DelActionType": kind, "DlyDelFlg": "N", "DlyPrcFlg": "TR",
            "DlyRet": 0.0 if returns is None else returns, "DlyOpen": price, "DlyHigh": price, "DlyLow": price,
            "DlyClose": price, "DlyVol": 1000.0, "DlyCap": cap, "DlyCumFacShr": 1.0,
        }
    )
    return frame.loc[:, list(COLUMNS)]


def delisting_row(permno, day, ret):
    row = rows_for(permno, [day], np.nan, np.nan)
    row["DlyDelFlg"], row["DlyPrcFlg"], row["DlyRet"], row["PrimaryExch"] = "Y", "DA", ret, "X"
    row["SecurityType"] = row["ShareType"] = row["DelActionType"] = None  # as on CRSP's delisting rows
    return row


class CrspUniverseTest(unittest.TestCase):
    def setUp(self):
        dividend = np.zeros(len(DAYS))
        dividend[5] = 0.02  # a 2% dividend: raw price flat, total return up
        etfs = [rows_for(p, DAYS, 100.0, 1e6).assign(SecurityType="FUND", SecuritySubType="ETF") for p in ETF_PERMNOS]
        a = rows_for(10001, DAYS, 10.0, 300.0, returns=dividend)
        self.rows = pd.concat(
            etfs
            + [
                a,
                a.iloc[[3]],  # a second row for a same-day distribution
                rows_for(10002, DAYS, 20.0, 200.0),
                rows_for(10003, DAYS, 4.0, 900.0),  # largest, but below the $5 minimum
                rows_for(10004, DAYS[:7], 30.0, 250.0, kind="MER"),
                delisting_row(10004, DAYS[7], 0.10),  # takeover premium paid in cash
                rows_for(10005, DAYS[:8], 6.0, 50.0, exchange="Q", kind="GDR"),
                delisting_row(10005, DAYS[8], np.nan),  # failed, no delisting return recorded
            ],
            ignore_index=True,
        )
        self.tables, self.counts = build_crsp_tables(self.rows, DAYS[0], DAYS[-1], top_n=2, min_price=5.0)

    def test_membership_is_top_market_cap_above_min_price_on_each_day(self):
        members = self.tables["membership"].groupby("trade_date")["sec_code"].apply(set)
        self.assertEqual(members[DAYS[0]], {"10001.CRSP", "10004.CRSP"})
        self.assertEqual(members[DAYS[9]], {"10001.CRSP", "10002.CRSP"})

    def test_prices_are_total_return_adjusted_and_distribution_rows_collapse(self):
        stock = self.tables["daily_stock"].set_index(["sec_code", "trade_date"]).sort_index()
        a = stock.loc["10001.CRSP"]
        self.assertEqual(len(a), len(DAYS))
        self.assertAlmostEqual(a.loc[DAYS[5], "close"] / a.loc[DAYS[4], "close"], 1.02)
        self.assertAlmostEqual(a.loc[DAYS[5], "adj_open"] / a.loc[DAYS[4], "adj_open"], 1.02)
        self.assertEqual(a.loc[DAYS[5], "raw_close"], a.loc[DAYS[4], "raw_close"])

    def test_delisting_exits_use_crsp_returns_and_shumway_only_when_missing(self):
        exits = self.tables["delistings"].set_index("sec_code")
        self.assertEqual(exits.loc["10004.CRSP", "kind"], "merger")
        self.assertAlmostEqual(exits.loc["10004.CRSP", "exit_factor"], 1.10)
        self.assertEqual(exits.loc["10004.CRSP", "last_trade_date"], DAYS[6])
        self.assertEqual(exits.loc["10005.CRSP", "kind"], "performance")
        self.assertAlmostEqual(exits.loc["10005.CRSP", "exit_factor"], 0.45)
        self.assertEqual(self.counts["delistings_with_crsp_return"], 1)

    def test_market_tables_hold_the_etfs_on_every_trading_day(self):
        index = self.tables["daily_index_us_market84"]
        self.assertEqual(sorted(index["index_code"].unique()), ["IJH.US", "IJR.US", "IWM.US", "SPY.US"])
        self.assertEqual(len(index), 4 * len(DAYS))


if __name__ == "__main__":
    unittest.main()

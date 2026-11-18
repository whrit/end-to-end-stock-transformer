"""Fake-client tests for the US Alpaca/OpenFIGI fetch and canonical table assembly."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    validate_calendar,
    validate_daily_index,
    validate_daily_stock,
    validate_membership,
)
from market_guided_stock_alpha.us_data import alpaca as us_alpaca
from market_guided_stock_alpha.us_data.openfigi import (
    MAX_JOBS_PER_REQUEST,
    MAX_REQUESTS_PER_MINUTE,
    OpenFigiError,
    map_tickers,
)
from market_guided_stock_alpha.us_data.universe import (
    MARKET84_ETFS,
    TABLE_FILES,
    build_securities,
    build_us_tables,
)


HAS_ALPACA = importlib.util.find_spec("alpaca") is not None
HAS_PARQUET = importlib.util.find_spec("pyarrow") is not None
FIGI = {
    "AAA": "BBG000AAAAA1",
    "BBB": "BBG000BBBBB2",
    "CCC": "BBG000CCCCC3",
    "DDD": "BBG000DDDDD4",
    "PFD": "BBG000PFDPF5",
    "ADR": "BBG000ADRAD6",
    "SPY": "BBG000SPYSP7",
    "IJH": "BBG000IJHIJ8",
    "IJR": "BBG000IJRIJ9",
    "IWM": "BBG000IWMIW0",
}


# ---------------------------------------------------------------- OpenFIGI


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.body


class FakeOpener:
    """Answers each job from a ticker table; ``failures`` 429s come first."""

    def __init__(self, answers, failures=0):
        self.answers = answers
        self.failures = failures
        self.requests = []

    def __call__(self, request, timeout=None):
        jobs = json.loads(request.data.decode("utf-8"))
        self.requests.append((request, jobs))
        if self.failures:
            self.failures -= 1
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"")
            )
        return FakeResponse(
            [
                self.answers.get(job["idValue"], {"warning": "No identifier found."})
                for job in jobs
            ]
        )


def figi_answer(figi, security_type2="Common Stock", exch="US", name="NAME"):
    return {
        "data": [
            {
                "figi": "BBG00XXXXXX0",
                "shareClassFIGI": figi,
                "compositeFIGI": "BBG00YYYYYY0",
                "securityType": "Common Stock",
                "securityType2": security_type2,
                "marketSector": "Equity",
                "ticker": "T",
                "name": name,
                "exchCode": exch,
            }
        ]
    }


class OpenFigiTest(unittest.TestCase):
    def test_batches_throttles_and_keeps_input_order(self):
        tickers = ["T{}".format(index) for index in range(250)]
        answers = {ticker: figi_answer(FIGI["AAA"]) for ticker in tickers[::2]}
        opener = FakeOpener(answers)
        sleeps = []

        result = map_tickers(tickers, "key", opener=opener, sleep=sleeps.append)

        self.assertEqual([len(jobs) for _, jobs in opener.requests], [100, 100, 50])
        self.assertEqual(MAX_JOBS_PER_REQUEST, 100)
        self.assertEqual(sleeps, [60.0 / MAX_REQUESTS_PER_MINUTE] * 2)
        self.assertLessEqual(60.0 / sleeps[0], 250)
        self.assertEqual(result["ticker"].tolist(), tickers)
        self.assertEqual(result["share_class_figi"].notna().sum(), 125)
        request, jobs = opener.requests[0]
        self.assertEqual(request.get_header("X-openfigi-apikey"), "key")
        self.assertEqual(
            jobs[0], {"idType": "TICKER", "idValue": "T0", "exchCode": "US"}
        )

    def test_share_class_ticker_uses_openfigi_slash(self):
        opener = FakeOpener({"BRK/B": figi_answer(FIGI["BBB"])})
        result = map_tickers(["BRK.B"], "key", opener=opener, sleep=lambda _: None)
        self.assertEqual(opener.requests[0][1][0]["idValue"], "BRK/B")
        self.assertEqual(result.loc[0, "ticker"], "BRK.B")
        self.assertEqual(result.loc[0, "share_class_figi"], FIGI["BBB"])

    def test_retries_http_429_with_backoff(self):
        opener = FakeOpener({"AAA": figi_answer(FIGI["AAA"])}, failures=2)
        sleeps = []
        result = map_tickers(["AAA"], "key", opener=opener, sleep=sleeps.append)
        self.assertEqual(len(opener.requests), 3)
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(result.loc[0, "share_class_figi"], FIGI["AAA"])

    def test_persistent_429_raises(self):
        opener = FakeOpener({}, failures=100)
        with self.assertRaisesRegex(OpenFigiError, "HTTP 429"):
            map_tickers(["AAA"], "key", opener=opener, sleep=lambda _: None)

    def test_unmapped_error_and_us_composite_choice(self):
        several = figi_answer(FIGI["CCC"], name="US LINE")
        several["data"].insert(
            0, dict(figi_answer(FIGI["DDD"], exch="LN")["data"][0], name="LONDON")
        )
        opener = FakeOpener(
            {
                "AAA": {"warning": "No identifier found."},
                "BAD": {"error": "Invalid idValue format"},
                "CCC": several,
            }
        )
        result = map_tickers(
            ["AAA", "BAD", "CCC"], "key", opener=opener, sleep=lambda _: None
        ).set_index("ticker")
        self.assertTrue(result.loc[["AAA", "BAD"], "share_class_figi"].isna().all())
        self.assertIn("No identifier found", result.loc["AAA", "mapping_note"])
        self.assertIn("error", result.loc["BAD", "mapping_note"])
        self.assertEqual(result.loc["CCC", "share_class_figi"], FIGI["CCC"])
        self.assertEqual(result.loc["CCC", "name"], "US LINE")

    def test_conflicting_us_share_class_figis_raise(self):
        conflict = figi_answer(FIGI["AAA"])
        conflict["data"].append(figi_answer(FIGI["BBB"])["data"][0])
        opener = FakeOpener({"AAA": conflict})
        with self.assertRaisesRegex(OpenFigiError, "several US share-class FIGIs"):
            map_tickers(["AAA"], "key", opener=opener, sleep=lambda _: None)


# ---------------------------------------------------------------- Alpaca


def barset_df(rows):
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "vwap"):
        if column not in frame.columns:
            frame[column] = 10.0
    frame["trade_count"] = 1.0
    return frame.set_index(["symbol", "timestamp"])


class AlpacaTest(unittest.TestCase):
    def test_bar_timestamps_map_to_new_york_session_dates(self):
        table = us_alpaca.bars_frame_to_table(
            barset_df(
                [
                    {"symbol": "aaa", "timestamp": "2024-03-08T05:00:00Z", "volume": 1},
                    {"symbol": "aaa", "timestamp": "2024-03-11T04:00:00Z", "volume": 2},
                    {"symbol": "bbb", "timestamp": "2024-03-12T03:00:00Z", "volume": 3},
                ]
            )
        )
        self.assertEqual(table["alpaca_symbol"].tolist(), ["AAA", "AAA", "BBB"])
        self.assertEqual(
            [value.date().isoformat() for value in table["trade_date"]],
            ["2024-03-08", "2024-03-11", "2024-03-11"],
        )
        self.assertEqual(str(table["trade_date"].dtype), "datetime64[ns]")

    def test_duplicate_session_bars_raise(self):
        frame = barset_df(
            [
                {"symbol": "AAA", "timestamp": "2024-03-08T05:00:00Z", "volume": 1},
                {"symbol": "AAA", "timestamp": "2024-03-08T15:00:00Z", "volume": 1},
            ]
        )
        with self.assertRaisesRegex(DataQualityError, "several daily bars for AAA"):
            us_alpaca.bars_frame_to_table(frame)

    def test_empty_barset_gives_empty_table(self):
        table = us_alpaca.bars_frame_to_table(pd.DataFrame())
        self.assertEqual(tuple(table.columns), us_alpaca.BAR_COLUMNS)

    def test_last_complete_date_waits_for_after_hours(self):
        self.assertEqual(
            us_alpaca.last_complete_date(pd.Timestamp("2024-03-12T23:59:00Z")),
            pd.Timestamp("2024-03-11"),
        )
        self.assertEqual(
            us_alpaca.last_complete_date(pd.Timestamp("2024-03-13T00:30:00Z")),
            pd.Timestamp("2024-03-12"),
        )

    @unittest.skipUnless(HAS_ALPACA, "alpaca-py is not installed")
    def test_fetch_daily_bars_chunks_symbols_and_requests_sip(self):
        calls = []

        class FakeData:
            def get_stock_bars(self, request):
                calls.append(request)
                return SimpleNamespace(
                    df=barset_df(
                        [
                            {
                                "symbol": symbol,
                                "timestamp": "2024-03-08T05:00:00Z",
                                "volume": 5,
                            }
                            for symbol in request.symbol_or_symbols
                        ]
                    )
                )

        table = us_alpaca.fetch_daily_bars(
            FakeData(), ["CCC", "AAA", "BBB", "AAA"], "2024-03-01", "2024-03-08",
            "raw", chunk_size=2, now=pd.Timestamp("2024-04-01T00:00:00Z"),
        )
        self.assertEqual(
            [request.symbol_or_symbols for request in calls], [["AAA", "BBB"], ["CCC"]]
        )
        self.assertEqual(calls[0].adjustment.value, "raw")
        self.assertEqual(calls[0].feed.value, "sip")
        # Request bounds are New York midnight..end-of-day, sent as naive UTC.
        self.assertEqual(calls[0].start, pd.Timestamp("2024-03-01T05:00:00").to_pydatetime())
        self.assertEqual(calls[0].end, pd.Timestamp("2024-03-09T04:59:59").to_pydatetime())
        self.assertEqual(sorted(table["alpaca_symbol"]), ["AAA", "BBB", "CCC"])

    @unittest.skipUnless(HAS_ALPACA, "alpaca-py is not installed")
    def test_list_assets_keeps_inactive_and_major_exchanges(self):
        statuses = []

        class FakeTrading:
            def get_all_assets(self, request):
                status = request.status.value
                statuses.append(status)
                return [
                    SimpleNamespace(
                        id="id-{}-{}".format(status, exchange),
                        symbol="X{}".format(exchange),
                        name="Name",
                        status=request.status,
                        exchange=exchange,
                        tradable=status == "active",
                    )
                    for exchange in ("NYSE", "OTC", "BATS")
                ]

        assets = us_alpaca.list_assets(FakeTrading())
        self.assertEqual(statuses, ["active", "inactive"])
        self.assertEqual(set(assets["exchange"]), {"NYSE", "BATS"})
        self.assertEqual(sorted(set(assets["status"])), ["active", "inactive"])
        self.assertEqual(assets["tradable"].tolist(), [True, True, False, False])

    @unittest.skipUnless(HAS_ALPACA, "alpaca-py is not installed")
    def test_fetch_trading_days(self):
        class FakeTrading:
            def get_calendar(self, request):
                self.request = request
                return [
                    SimpleNamespace(date=pd.Timestamp("2024-01-03").date()),
                    SimpleNamespace(date=pd.Timestamp("2024-01-02").date()),
                ]

        client = FakeTrading()
        days = us_alpaca.fetch_trading_days(client, "2024-01-01", "2024-01-03")
        self.assertEqual(days, [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
        self.assertEqual(str(client.request.start), "2024-01-01")


# ---------------------------------------------------------------- Universe

DAYS = list(pd.bdate_range("2024-01-02", periods=25))
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2024-02-06")


def assets_frame():
    symbols = list(FIGI) + ["UNM"]
    rows = [
        {
            "asset_id": "id-" + symbol,
            "alpaca_symbol": symbol,
            "name": symbol + " Inc",
            "status": "active",
            "exchange": "NYSE",
            "tradable": True,
        }
        for symbol in symbols
    ]
    rows.append(dict(rows[0], asset_id="id-old-AAA", status="inactive"))
    return pd.DataFrame(rows)


def figi_frame(overrides=None):
    types = {"PFD": "Preferred Stock", "ADR": "Depositary Receipt"}
    for etf in MARKET84_ETFS:
        types[etf] = "ETP"
    rows = []
    for symbol in list(FIGI) + ["UNM"]:
        mapped = symbol != "UNM"
        rows.append(
            {
                "ticker": symbol,
                "share_class_figi": FIGI.get(symbol) if mapped else None,
                "security_type2": types.get(symbol, "Common Stock") if mapped else None,
                "market_sector": "Equity" if mapped else None,
                "name": symbol if mapped else None,
                "mapping_note": "mapped" if mapped else "warning: No identifier found.",
            }
        )
    frame = pd.DataFrame(rows)
    for symbol, figi in (overrides or {}).items():
        frame.loc[frame["ticker"] == symbol, "share_class_figi"] = figi
    return frame


def bar_spec():
    """symbol -> (raw close, raw volume per day index, split factor, missing indices)."""
    return {
        # Exactly $5 and exactly $5M dollar volume: member from the 20th day.
        "AAA": (5.0, lambda i: 1000000.0, 0.5, ()),
        # Just below $5 with huge volume: never a member.
        "BBB": (4.99, lambda i: 1.0e8, 1.0, ()),
        # $10M dollar volume from day 12: the 20-day mean (day t included)
        # first reaches $5M on day 21 (days 12..21).
        "CCC": (10.0, lambda i: 1000000.0 if i >= 12 else 0.0, 1.0, ()),
        # Missing bar on day 3: windows touching it are incomplete.
        "DDD": (20.0, lambda i: 1000000.0, 1.0, (3,)),
        "PFD": (25.0, lambda i: 1.0e7, 1.0, ()),
        "ADR": (30.0, lambda i: 1.0e7, 1.0, ()),
        "UNM": (40.0, lambda i: 1.0e7, 1.0, ()),
        "SPY": (400.0, lambda i: 1000.0 + i, 1.0, ()),
        "IJH": (50.0, lambda i: 2000.0, 1.0, ()),
        "IJR": (100.0, lambda i: 3000.0, 1.0, ()),
        "IWM": (200.0, lambda i: 4000.0, 1.0, ()),
    }


def bar_frames(spec=None):
    adjusted, raw = [], []
    for symbol, (close, volume, factor, missing) in (spec or bar_spec()).items():
        for index, day in enumerate(DAYS):
            if index in missing:
                continue
            raw_close = close + 0.01 * index if symbol in MARKET84_ETFS else close
            base = {"alpaca_symbol": symbol, "trade_date": day}
            raw.append(
                dict(
                    base, open=raw_close, high=raw_close, low=raw_close,
                    close=raw_close, volume=volume(index), vwap=raw_close + 1.0,
                )
            )
            adjusted.append(
                dict(
                    base, open=raw_close * factor * 0.9, high=raw_close * factor,
                    low=raw_close * factor, close=raw_close * factor,
                    volume=volume(index) / factor, vwap=(raw_close + 1.0) * factor,
                )
            )
    return pd.DataFrame(adjusted), pd.DataFrame(raw)


def build(securities=None, adjusted=None, raw=None, days=None):
    if securities is None:
        securities, _ = build_securities(assets_frame(), figi_frame())
    default_adjusted, default_raw = bar_frames()
    return build_us_tables(
        securities,
        default_adjusted if adjusted is None else adjusted,
        default_raw if raw is None else raw,
        DAYS if days is None else days,
        START,
        END,
    )


class UniverseTest(unittest.TestCase):
    def test_securities_keep_mapped_common_stocks_and_count_exclusions(self):
        securities, counts = build_securities(assets_frame(), figi_frame())
        self.assertEqual(sorted(securities["alpaca_symbol"]), ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(
            tuple(securities.columns),
            ("sec_code", "ticker", "alpaca_symbol", "name", "status", "exchange",
             "security_type2"),
        )
        self.assertEqual(
            securities.set_index("alpaca_symbol").loc["AAA", "status"], "active"
        )
        self.assertEqual(
            counts,
            {
                "assets_seen": 12,
                "assets_active": 11,
                "assets_inactive": 1,
                "duplicate_symbol_rows": 1,
                "symbols": 11,
                "mapped": 10,
                "unmapped": 1,
                "excluded_by_type": 6,
                "common_stocks": 4,
            },
        )

    def test_missing_openfigi_row_is_rejected(self):
        figi = figi_frame()
        figi = figi.loc[figi["ticker"] != "BBB"]
        with self.assertRaisesRegex(DataQualityError, "lacks rows for Alpaca symbols: BBB"):
            build_securities(assets_frame(), figi)

    def test_daily_stock_has_adjusted_and_raw_columns_keyed_by_figi(self):
        stock = build()["daily_stock"]
        self.assertEqual(
            tuple(stock.columns),
            ("trade_date", "sec_code", "open", "high", "low", "close", "vwap",
             "volume", "adj_open", "raw_close", "raw_volume"),
        )
        self.assertEqual(
            set(stock["sec_code"]),
            {FIGI["AAA"], FIGI["BBB"], FIGI["CCC"], FIGI["DDD"]},
        )
        aaa = stock.loc[stock["sec_code"] == FIGI["AAA"]].iloc[0]
        self.assertAlmostEqual(aaa["close"], 2.5)
        self.assertAlmostEqual(aaa["open"], 2.25)
        self.assertAlmostEqual(aaa["adj_open"], 2.25)
        self.assertAlmostEqual(aaa["volume"], 2000000.0)
        self.assertAlmostEqual(aaa["raw_close"], 5.0)
        self.assertAlmostEqual(aaa["raw_volume"], 1000000.0)
        self.assertEqual(len(stock), 25 * 4 - 1)

    def test_etf_index_tables_use_adjusted_prices_and_raw_amount(self):
        tables = build()
        market63 = tables["daily_index_us_market63"]
        market84 = tables["daily_index_us_market84"]
        self.assertEqual(sorted(set(market63["index_code"])), ["IJH.US", "IJR.US", "SPY.US"])
        self.assertEqual(
            sorted(set(market84["index_code"])), ["IJH.US", "IJR.US", "IWM.US", "SPY.US"]
        )
        spy = market84.loc[market84["index_code"] == "SPY.US"].reset_index(drop=True)
        self.assertEqual(len(spy), 25)
        day = 4
        raw_close = 400.0 + 0.01 * day
        self.assertAlmostEqual(spy.loc[day, "close"], raw_close)
        self.assertAlmostEqual(spy.loc[day, "open"], raw_close * 0.9)
        self.assertAlmostEqual(spy.loc[day, "amount"], (1000.0 + day) * (raw_close + 1.0))

    def test_membership_filter_edges(self):
        membership = build()["membership"]
        first = membership.groupby("sec_code")["trade_date"].min()
        counts = membership.groupby("sec_code").size()
        self.assertEqual(first[FIGI["AAA"]], DAYS[19])
        self.assertEqual(counts[FIGI["AAA"]], 6)
        self.assertNotIn(FIGI["BBB"], set(membership["sec_code"]))
        self.assertEqual(first[FIGI["CCC"]], DAYS[21])
        self.assertEqual(first[FIGI["DDD"]], DAYS[23])
        self.assertEqual(
            set(membership["sec_code"]), {FIGI["AAA"], FIGI["CCC"], FIGI["DDD"]}
        )

    def test_calendar_covers_every_day(self):
        calendar = build()["calendar"]
        self.assertEqual(len(calendar), (END - START).days + 1)
        self.assertEqual(calendar["calendar_date"].iloc[0], START)
        self.assertEqual(int(calendar["is_open"].sum()), 25)
        self.assertEqual(int(calendar.loc[0, "is_open"]), 0)

    def test_outputs_are_validator_clean(self):
        tables = build()
        self.assertEqual(set(tables), set(TABLE_FILES))
        stock = tables["daily_stock"]
        pd.testing.assert_frame_equal(
            validate_daily_stock(stock, require_adj_open=True),
            stock.drop(columns=["raw_close", "raw_volume"]),
        )
        for name in ("daily_index_us_market63", "daily_index_us_market84"):
            pd.testing.assert_frame_equal(validate_daily_index(tables[name]), tables[name])
        pd.testing.assert_frame_equal(
            validate_membership(tables["membership"]), tables["membership"]
        )
        pd.testing.assert_frame_equal(
            validate_calendar(tables["calendar"]), tables["calendar"]
        )

    def test_duplicate_figi_on_same_date_is_rejected(self):
        assets = assets_frame()
        assets = pd.concat(
            [assets, assets.iloc[[0]].assign(asset_id="id-AAX", alpaca_symbol="AAX")],
            ignore_index=True,
        )
        figi = pd.concat(
            [figi_frame(), figi_frame().iloc[[0]].assign(ticker="AAX")],
            ignore_index=True,
        )
        securities, _ = build_securities(assets, figi)
        spec = bar_spec()
        spec["AAX"] = spec["AAA"]
        adjusted, raw = bar_frames(spec)
        with self.assertRaisesRegex(
            DataQualityError,
            "share-class FIGI BBG000AAAAA1 maps to several Alpaca symbols on "
            "2024-01-02: AAA,AAX",
        ):
            build(securities, adjusted, raw)

    def test_adjusted_and_raw_coverage_must_match(self):
        adjusted, raw = bar_frames()
        raw = raw.iloc[1:]
        with self.assertRaisesRegex(
            DataQualityError, "AAA on 2024-01-02 exists only in adjusted bars"
        ):
            build(adjusted=adjusted, raw=raw)

    def test_bar_on_closed_day_is_rejected(self):
        with self.assertRaisesRegex(DataQualityError, "not an open trading day"):
            build(days=DAYS[1:])

    def test_etf_missing_session_is_rejected(self):
        spec = bar_spec()
        close, volume, factor, _ = spec["IWM"]
        spec["IWM"] = (close, volume, factor, (7,))
        adjusted, raw = bar_frames(spec)
        with self.assertRaisesRegex(DataQualityError, "ETF IWM lacks Alpaca bars on 1"):
            build(adjusted=adjusted, raw=raw)


# ---------------------------------------------------------------- Script


def load_fetch_script():
    path = ROOT / "scripts" / "fetch_us_daily.py"
    spec = importlib.util.spec_from_file_location("fetch_us_daily_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FetchScriptTest(unittest.TestCase):
    def test_missing_environment_variables_are_named(self):
        script = load_fetch_script()
        with self.assertRaisesRegex(
            SystemExit, "APCA_API_SECRET_KEY, OPENFIGI_API_KEY"
        ):
            script.required_env({"APCA_API_KEY_ID": "x"})

    @unittest.skipUnless(HAS_ALPACA and HAS_PARQUET, "needs alpaca-py and pyarrow")
    def test_main_writes_tables_and_manifest_to_new_local_dir(self):
        script = load_fetch_script()
        adjusted, raw = bar_frames()
        env = {
            "APCA_API_KEY_ID": "id",
            "APCA_API_SECRET_KEY": "secret",
            "OPENFIGI_API_KEY": "figi",
        }

        def fake_bars(client, symbols, start, end, adjustment):
            frame = adjusted if adjustment == "all" else raw
            return frame.loc[frame["alpaca_symbol"].isin(symbols)]

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", env), \
                mock.patch("alpaca.trading.client.TradingClient"), \
                mock.patch("alpaca.data.historical.StockHistoricalDataClient"), \
                mock.patch.object(script, "list_assets", return_value=assets_frame()), \
                mock.patch.object(script, "map_tickers", return_value=figi_frame()), \
                mock.patch.object(script, "fetch_daily_bars", side_effect=fake_bars), \
                mock.patch.object(script, "fetch_trading_days", return_value=DAYS), \
                mock.patch.object(
                    script, "last_complete_date", return_value=pd.Timestamp("2024-12-31")
                ), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            output = Path(tmp) / "us" / "raw"
            self.assertEqual(
                script.main(
                    ["--start", str(START.date()), "--end", str(END.date()),
                     "--output", str(output)]
                ),
                0,
            )
            written = sorted(path.name for path in output.iterdir())
            manifest = json.loads((output / "fetch_manifest.json").read_text())
            stock = pd.read_parquet(output / "daily_stock.parquet")
            with self.assertRaisesRegex(SystemExit, "output already exists"):
                script.main(["--output", str(output)])
        self.assertEqual(
            written, sorted(list(TABLE_FILES.values()) + ["fetch_manifest.json"])
        )
        self.assertEqual(manifest["counts"]["unmapped"], 1)
        self.assertEqual(manifest["counts"]["excluded_by_type"], 6)
        self.assertEqual(manifest["end"], "2024-02-06")
        self.assertEqual(manifest["tables"]["daily_stock"]["rows"], len(stock))
        self.assertNotIn("secret", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()

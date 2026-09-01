"""Focused canonical model and transport contracts."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.dataset import FeatureLayout, SequenceSample
from market_guided_stock_alpha.modeling import (
    GRURegressor,
    LightGBMRegressor,
    MasterConfig,
    MasterModel,
    VanillaTransformerRegressor,
    adapt_sequence_sample,
    flatten_sequence_sample,
    forward_date_batch,
    lightgbm_feature_order,
)
from market_guided_stock_alpha.modeling.experimental import (
    DualFrequencyMasterConfig,
    DualFrequencyMasterModel,
)


def sample(layout, security_count=4):
    dates = pd.bdate_range("2024-01-02", periods=layout.lookback)
    stock = np.arange(
        security_count * layout.lookback * layout.stock_dim, dtype="float32"
    ).reshape(security_count, layout.lookback, layout.stock_dim)
    market = np.arange(
        layout.lookback * layout.market_dim, dtype="float32"
    ).reshape(layout.lookback, layout.market_dim)
    intraday_market = (
        np.zeros((layout.lookback, layout.intraday_market_dim), dtype="float32")
        if layout.intraday_market_dim
        else None
    )
    return SequenceSample(
        prediction_date=dates[-1],
        history_dates=dates,
        security_ids=np.arange(security_count, dtype="int32"),
        security_codes=tuple("S{:03d}".format(i) for i in range(security_count)),
        stock_features=stock,
        market_features=market,
        labels_raw=np.linspace(-0.02, 0.03, security_count),
        labels=np.linspace(-1.0, 1.0, security_count).astype("float32"),
        label_mask=np.ones(security_count, dtype="bool"),
        stock_mask=np.ones((security_count, layout.lookback), dtype="bool"),
        component_masks={},
        market_mask=np.ones(layout.lookback, dtype="bool"),
        metadata={},
        intraday_market_features=intraday_market,
        intraday_market_mask=(
            np.ones(layout.lookback, dtype="bool")
            if intraday_market is not None
            else None
        ),
    )


class MasterConfigurationTest(unittest.TestCase):
    def test_historical_and_non_whitelist_dimensions_validate(self):
        for d_model in (128, 256, 512, 384):
            with self.subTest(d_model=d_model):
                config = MasterConfig(158, 63, d_model=d_model)
                self.assertEqual(config.d_model, d_model)

    def test_semantic_validation(self):
        with self.assertRaisesRegex(ValueError, "temporal_heads"):
            MasterConfig(158, 63, d_model=250, temporal_heads=4)
        with self.assertRaisesRegex(ValueError, "dropout"):
            MasterConfig(158, 63, dropout=1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            MasterConfig(158, 63, beta=0.0)

    def test_stage_layout_forward_shapes(self):
        layouts = (
            FeatureLayout.stage1(),
            FeatureLayout.stage2(),
            FeatureLayout.stage2("market84"),
        )
        expected = ((158, 63), (406, 63), (406, 84))
        for layout, dimensions in zip(layouts, expected):
            with self.subTest(position=layout.position):
                self.assertEqual((layout.stock_dim, layout.market_dim), dimensions)
                model = MasterModel.from_layout(
                    layout, d_model=16, temporal_heads=4, spatial_heads=2, dropout=0.0
                )
                batch = adapt_sequence_sample(sample(layout))
                output = forward_date_batch(model, batch)
                self.assertEqual(tuple(output.shape), (4,))

    def test_formal_ablation_modes_forward(self):
        layout = FeatureLayout.stage1()
        batch = adapt_sequence_sample(sample(layout, 3))
        for mode in ("no_market", "no_intra", "no_inter", "no_temporal"):
            with self.subTest(mode=mode):
                model = MasterModel.from_layout(
                    layout, d_model=16, dropout=0.0, ablation=mode
                )
                self.assertEqual(tuple(forward_date_batch(model, batch).shape), (3,))


class BaselineAndExperimentalTest(unittest.TestCase):
    def test_gru_transformer_stock_sequence_forward(self):
        layout = FeatureLayout.stage2()
        batch = adapt_sequence_sample(sample(layout, 3))
        gru = GRURegressor.from_layout(
            layout, hidden_dim=8, num_layers=1, dropout=0.0
        )
        transformer = VanillaTransformerRegressor.from_layout(
            layout,
            d_model=8,
            num_layers=1,
            heads=2,
            dropout=0.0,
            dim_feedforward=16,
        )
        for model in (gru, transformer):
            output = forward_date_batch(model, batch)
            self.assertEqual(tuple(output.shape), (3,))
            output.square().mean().backward()

    def test_lightgbm_uses_full_eight_day_flatten(self):
        layout = FeatureLayout.stage1()
        value = sample(layout, 2)
        flat = flatten_sequence_sample(value)
        self.assertEqual(
            flat.shape,
            (2, layout.lookback * (layout.stock_dim + layout.market_dim)),
        )
        self.assertEqual(len(lightgbm_feature_order(layout)), flat.shape[1])
        self.assertEqual(flat[0, 0], value.stock_features[0, 0, 0])
        anchor_offset = (layout.lookback - 1) * (
            layout.stock_dim + layout.market_dim
        )
        self.assertEqual(
            flat[0, anchor_offset], value.stock_features[0, -1, 0]
        )

    @unittest.skipUnless(
        importlib.util.find_spec("lightgbm") is not None,
        "optional lightgbm package is not installed",
    )
    def test_lightgbm_model_smoke_on_historical_representation(self):
        layout = FeatureLayout.stage1()
        rows = []
        labels = []
        for position in range(4):
            value = sample(layout, 3)
            value.stock_features[:, :, 0] += position
            rows.append(flatten_sequence_sample(value))
            labels.extend([float(position), float(position + 1), float(position + 2)])
        features = np.concatenate(rows, axis=0)
        model = LightGBMRegressor().fit(
            features,
            np.asarray(labels),
            {
                "objective": "regression",
                "learning_rate": 0.1,
                "num_leaves": 4,
                "min_data_in_leaf": 1,
                "verbose": -1,
                "num_threads": 1,
            },
            num_boost_round=2,
        )
        predictions = model.predict(features)
        self.assertEqual(predictions.shape, (12,))
        self.assertTrue(np.isfinite(predictions).all())

    def test_dual_frequency_forward_and_non_whitelist_dimension(self):
        layout = FeatureLayout.dual_frequency()
        config = DualFrequencyMasterConfig.from_layout(
            layout,
            d_model=24,
            temporal_heads=3,
            spatial_heads=2,
            dropout=0.0,
        )
        model = DualFrequencyMasterModel(config)
        output = forward_date_batch(model, adapt_sequence_sample(sample(layout, 3)))
        self.assertEqual(tuple(output.shape), (3,))


if __name__ == "__main__":
    unittest.main()

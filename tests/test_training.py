"""Canonical losses, whole-date training, and checkpoint tests."""

import tempfile
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

from market_guided_stock_alpha.dataset import (
    DatasetStatistics,
    FeatureLayout,
    FeatureStatistics,
    SequenceSample,
)
from market_guided_stock_alpha.modeling import (
    GRURegressor,
    MasterModel,
    VanillaTransformerRegressor,
    WholeDateLoader,
)
from market_guided_stock_alpha.training import (
    CheckpointContract,
    CheckpointManager,
    EarlyStoppingConfig,
    LossConfig,
    OptimizerConfig,
    Trainer,
    TrainingConfig,
    build_loss,
    load_preprocessing,
    preprocessing_to_dict,
    save_preprocessing,
)


def identity_statistics(layout):
    families = {}
    names_by_family = {
        family: (
            layout.stock_feature_names[
                layout.family_slices[family][0] : layout.family_slices[family][1]
            ]
        )
        for family in layout.stock_families
    }
    names_by_family[layout.market_family] = layout.market_feature_names
    for family, names in names_by_family.items():
        count = len(names)
        families[family] = FeatureStatistics(
            feature_names=tuple(names),
            median=np.zeros(count, dtype="float64"),
            mad=np.ones(count, dtype="float64"),
            scale=np.ones(count, dtype="float64"),
            finite_count=np.ones(count, dtype="int64"),
            zero_mad_mask=np.zeros(count, dtype="bool"),
        )
    return DatasetStatistics(families)


def date_sample(layout, position, security_count=4):
    dates = pd.bdate_range("2024-01-02", periods=layout.lookback + position)
    rng = np.random.default_rng(100 + position)
    labels = np.asarray([-1.3416408, -0.4472136, 0.4472136, 1.3416408], dtype="float32")
    return SequenceSample(
        prediction_date=dates[-1],
        history_dates=dates[-layout.lookback :],
        security_ids=np.arange(security_count, dtype="int32"),
        security_codes=tuple("S{}".format(i) for i in range(security_count)),
        stock_features=rng.normal(
            size=(security_count, layout.lookback, layout.stock_dim)
        ).astype("float32"),
        market_features=rng.normal(
            size=(layout.lookback, layout.market_dim)
        ).astype("float32"),
        labels_raw=np.asarray([-0.02, -0.01, 0.01, 0.03], dtype="float64"),
        labels=labels,
        label_mask=np.ones(security_count, dtype="bool"),
        stock_mask=np.ones((security_count, layout.lookback), dtype="bool"),
        component_masks={},
        market_mask=np.ones(layout.lookback, dtype="bool"),
        metadata={},
    )


class TinyDataset:
    def __init__(self, layout, count):
        self.values = [date_sample(layout, position) for position in range(count)]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, position):
        return self.values[position]


class LossTest(unittest.TestCase):
    def test_loss_specific_defaults_preserve_historical_alpha(self):
        self.assertEqual(LossConfig(name="rpl").alpha, 0.5)
        self.assertEqual(LossConfig(name="rsr").alpha, 1.0)

    def test_all_formal_losses_mask_backward_and_shape_validation(self):
        for name in ("mse", "ccc", "listfold", "rpl", "rsr"):
            with self.subTest(name=name):
                predictions = torch.tensor(
                    [0.1, -0.2, 0.3, 0.0, 0.4], requires_grad=True
                )
                labels = torch.tensor([0.2, -0.1, 0.1, 0.5, 0.3])
                mask = torch.tensor([True, True, False, True, True])
                config = LossConfig(
                    name=name,
                    minimum_count=2,
                    pairwise_chunk_size=2,
                )
                result = build_loss(config)(predictions, labels, mask, labels)
                self.assertEqual(result["valid_count"], 4)
                self.assertTrue(bool(torch.isfinite(result["loss"])))
                result["loss"].backward()
                self.assertIsNotNone(predictions.grad)
                self.assertTrue(bool(torch.isfinite(predictions.grad).all()))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            build_loss("mse")(
                torch.zeros((2, 1)),
                torch.zeros((2, 1)),
                torch.ones((2, 1), dtype=torch.bool),
            )


class TrainingAndCheckpointTest(unittest.TestCase):
    def test_master_training_checkpoint_reload_and_preprocessing(self):
        layout = FeatureLayout.stage1()
        statistics = identity_statistics(layout)
        preprocessing = preprocessing_to_dict(statistics)
        config = TrainingConfig(
            epochs=1,
            optimizer=OptimizerConfig(
                learning_rate=1e-3, gradient_clip_value=3.0
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=False, patience=1, minimum_epochs=1
            ),
            seed=7,
        )
        model = MasterModel.from_layout(
            layout, d_model=16, temporal_heads=4, spatial_heads=2, dropout=0.0
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = CheckpointContract.from_components(
                model, config, layout, preprocessing
            )
            manager = CheckpointManager(root / "checkpoints", contract)
            trainer = Trainer(model, config, device="cpu", checkpoint_manager=manager)
            result = trainer.fit(
                WholeDateLoader(TinyDataset(layout, 2), shuffle=True, seed=7),
                WholeDateLoader(TinyDataset(layout, 1)),
            )
            self.assertEqual(result["best_epoch"], 1)
            self.assertTrue((root / "checkpoints" / "last.pt").is_file())
            self.assertTrue((root / "checkpoints" / "best.pt").is_file())

            reloaded = MasterModel.from_layout(
                layout, d_model=16, temporal_heads=4, spatial_heads=2, dropout=0.0
            )
            reload_trainer = Trainer(reloaded, config, device="cpu")
            payload = manager.load(
                root / "checkpoints" / "last.pt",
                reloaded,
                reload_trainer.optimizer,
                restore_rng=False,
            )
            self.assertEqual(payload["contract"]["model_config"], model.config_dict())
            self.assertEqual(
                payload["contract"]["preprocessing_state"]["fit_scope"],
                "training_dates_only",
            )

            preprocessing_path = root / "preprocessing.json"
            save_preprocessing(preprocessing_path, statistics)
            loaded = load_preprocessing(preprocessing_path)
            self.assertEqual(
                loaded.require("alpha158", layout.stock_feature_names).feature_names,
                layout.stock_feature_names,
            )

            incompatible = CheckpointContract.from_components(
                MasterModel.from_layout(
                    FeatureLayout.stage2(), d_model=16, dropout=0.0
                ),
                config,
                FeatureLayout.stage2(),
                preprocessing,
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                CheckpointManager(root / "other", incompatible).load(
                    root / "checkpoints" / "last.pt", reloaded, restore_rng=False
                )

    def test_gru_and_transformer_train_smoke(self):
        layout = FeatureLayout.stage1()
        config = TrainingConfig(
            epochs=1,
            early_stopping=EarlyStoppingConfig(
                enabled=False, patience=1, minimum_epochs=1
            ),
        )
        models = (
            GRURegressor.from_layout(
                layout, hidden_dim=8, num_layers=1, dropout=0.0
            ),
            VanillaTransformerRegressor.from_layout(
                layout,
                d_model=8,
                num_layers=1,
                heads=2,
                dropout=0.0,
                dim_feedforward=16,
            ),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                trainer = Trainer(model, config, device="cpu")
                output = trainer.train_epoch(
                    WholeDateLoader(TinyDataset(layout, 1)), epoch=1
                )
                self.assertTrue(np.isfinite(output["train_loss"]))


if __name__ == "__main__":
    unittest.main()

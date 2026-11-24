import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "remote_script", Path(__file__).resolve().parents[1] / "scripts" / "remote.py"
)
assert _SPEC is not None and _SPEC.loader is not None
remote = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(remote)


class BestValueTest(unittest.TestCase):
    def test_keeps_only_offers_faster_than_every_cheaper_one(self):
        offers = [
            {"id": "slow_pricey", "dph_total": 0.20, "total_flops": 10.0},
            {"id": "tie_slower", "dph_total": 0.10, "total_flops": 15.0},
            {"id": "cheap", "dph_total": 0.05, "total_flops": 12.0},
            {"id": "tie_faster", "dph_total": 0.10, "total_flops": 30.0},
            {"id": "same_speed_pricier", "dph_total": 0.30, "total_flops": 30.0},
            {"id": "fast", "dph_total": 0.40, "total_flops": 80.0},
        ]
        kept = [offer["id"] for offer in remote._best_value(offers)]
        self.assertEqual(kept, ["cheap", "tie_faster", "fast"])

    def test_ranks_multi_gpu_offers_by_the_whole_machine(self):
        # The agent runs jobs on every GPU, so two slower cards can beat one faster card.
        offers = [
            {"id": "one_fast", "dph_total": 0.10, "total_flops": 30.0, "num_gpus": 1},
            {"id": "two_slow", "dph_total": 0.15, "total_flops": 40.0, "num_gpus": 2},
            {"id": "four_slower", "dph_total": 0.20, "total_flops": 36.0, "num_gpus": 4},
        ]
        self.assertEqual([offer["id"] for offer in remote._best_value(offers)], ["one_fast", "two_slow"])


if __name__ == "__main__":
    unittest.main()

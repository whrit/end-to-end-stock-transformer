"""Formal Stage-1 MASTER ablations without copying the backbone."""

from typing import Any, Mapping

import torch


ABLATION_MODES = ("none", "no_market", "no_intra", "no_inter", "no_temporal")


def validate_ablation_mode(value: Any) -> str:
    mode = "none" if value is None else str(value)
    if mode not in ABLATION_MODES:
        raise ValueError(
            "MASTER ablation must be one of: {}".format(", ".join(ABLATION_MODES))
        )
    return mode


class _MASTERAblationMixin:
    def _ablation_source(self, values: torch.Tensor) -> torch.Tensor:
        source = values[:, :, : self.gate_input_start_index]
        if self.ablation_mode == "no_market":
            return source
        gate_input = values[
            :, -1, self.gate_input_start_index : self.gate_input_end_index
        ]
        return source * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        source = self._ablation_source(values)
        encoded = self.layers[1](self.layers[0](source))
        intra = encoded if self.ablation_mode == "no_intra" else self.layers[2](encoded)
        inter = intra if self.ablation_mode == "no_inter" else self.layers[3](intra)
        representation = (
            inter[:, -1, :]
            if self.ablation_mode == "no_temporal"
            else self.layers[4](inter)
        )
        return self.layers[5](representation).squeeze(-1)


def create_master_ablation(
    master_class: Any, architecture: Mapping[str, Any], ablation_mode: str
) -> Any:
    mode = validate_ablation_mode(ablation_mode)
    if mode == "none":
        raise ValueError("full MASTER does not require an ablation wrapper")

    class MASTERAblation(_MASTERAblationMixin, master_class):
        pass

    model = MASTERAblation(**dict(architecture))
    model.ablation_mode = mode
    return model

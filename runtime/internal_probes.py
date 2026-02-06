"""
Internal Probes (Stateless)
===========================
Probes that inspect instantaneous neural activation states.
"""
import torch
import torch.nn.functional as F
from typing import Any, Dict
from runtime.interface import ModelOutput


class LayerDeltaProbe:
    """
    Computes the structural change between the last two layers.
    Metric: layer_delta = 1.0 - cosine_similarity(last, prev)
    High delta => Reasoning jump or internal inconsistency.
    """

    def observe(self, model_raw_output: Any) -> ModelOutput:
        # Contract: Inputs are 1D Tensors [dim]
        h_last = model_raw_output.get("last_hidden_state")
        h_prev = model_raw_output.get("prev_hidden_state")

        if h_last is None or h_prev is None:
            return {}

        # Compute Cosine Similarity
        # unsqueeze(0) to make them [1, dim] for cosine_similarity
        cos_sim = F.cosine_similarity(h_last.unsqueeze(0), h_prev.unsqueeze(0)).item()

        # Delta: 0 (identical) to 2 (opposite)
        delta = 1.0 - cos_sim

        return {
            "layer_similarity": cos_sim,
            "layer_delta": delta
        }


class NormProbe:
    """
    Computes the energy (L2 norm) of the final activation.
    High norm => Potential OOD token or numerical instability.
    """

    def observe(self, model_raw_output: Any) -> ModelOutput:
        h_last = model_raw_output.get("last_hidden_state")

        if h_last is None:
            return {}

        norm_val = torch.norm(h_last, p=2).item()

        return {
            "activation_energy": norm_val
        }
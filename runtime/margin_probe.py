"""
Margin Probe
============
Computes confidence margin between top-1 and top-2 token probabilities.
Higher margin => higher confidence.
"""
from typing import Any
import torch
from runtime.interface import ModelOutput

class MarginProbe:
    def observe(self, model_raw_output: Any) -> ModelOutput:
        logits = model_raw_output["logits"]
        # Convert to tensor for calculation
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        top2 = torch.topk(probs, k=2)
        # Margin = p(best) - p(second_best)
        margin = (top2.values[0] - top2.values[1]).item()

        return {
            "margin": margin,
        }
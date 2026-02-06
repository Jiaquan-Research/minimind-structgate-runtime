"""
Token Entropy Probe (Real)
==========================
Compute Shannon entropy from the model's raw logits.
Operates on local epistemic uncertainty (next-token distribution).
"""
import math
import torch
from typing import Any
from runtime.interface import ModelOutput

class TokenEntropyProbe:
    def observe(self, model_raw_output: Any) -> ModelOutput:
        # Input: List[float]
        logits_list = model_raw_output["logits"]

        # Convert to Tensor for numerical stability
        logits_tensor = torch.tensor(logits_list)
        probs_tensor = torch.nn.functional.softmax(logits_tensor, dim=-1)

        # Entropy = -sum(p * log(p))
        # Add epsilon 1e-9 to prevent log(0)
        log_probs = torch.log(probs_tensor + 1e-9)
        entropy = -torch.sum(probs_tensor * log_probs).item()

        # Convert back to list
        probs = probs_tensor.tolist()

        return {
            "logits": logits_list,
            "token_probs": probs,
            "entropy": entropy,
        }
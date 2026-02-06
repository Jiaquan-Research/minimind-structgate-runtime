"""
Dummy Runtime Probe
===================

Extracts entropy-like signals from fake logits.
"""

import math
from typing import Any, List

from runtime.interface import ModelOutput


class DummyProbe:
    """
    Minimal probe computing categorical entropy.
    """

    def observe(self, model_raw_output: Any) -> ModelOutput:
        logits: List[float] = model_raw_output["logits"]

        # softmax
        exps = [math.exp(x) for x in logits]
        total = sum(exps)
        probs = [e / total for e in exps]

        # entropy
        entropy = -sum(p * math.log(p + 1e-8) for p in probs)

        return {
            "logits": logits,
            "token_probs": probs,
            "entropy": entropy,
        }

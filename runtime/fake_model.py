"""
Fake Language Model
===================

A deterministic, controllable stand-in for a real language model.

Purpose:
- Simulate uncertainty
- Produce predictable entropy patterns
- Validate runtime + StructGate wiring
"""

import random
from typing import Any, List


class FakeModel:
    """
    Fake language model producing synthetic logits.
    """

    def __init__(self, seed: int = 0):
        random.seed(seed)

    def forward(self, prompt: str) -> Any:
        """
        Generate fake logits based on prompt length.

        Short prompt -> low entropy
        Long / vague prompt -> high entropy
        """
        length = len(prompt)

        if length < 20:
            # confident output
            logits = [10.0, 1.0, 0.5]
        else:
            # uncertain output
            logits = [1.0, 1.0, 1.0]

        return {
            "logits": logits
        }

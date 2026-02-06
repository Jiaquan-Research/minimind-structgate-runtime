"""
Simple Entropy Threshold Policy
===============================

First admissibility policy for StructGate.
"""

from runtime.interface import ModelOutput
from structgate.decision import GateAction


class EntropyThresholdPolicy:
    """
    Refuse when uncertainty exceeds threshold.
    """

    def __init__(self, max_entropy: float):
        self.max_entropy = max_entropy

    def decide(self, obs: ModelOutput) -> GateAction:
        if obs["entropy"] > self.max_entropy:
            return GateAction.REFUSE
        return GateAction.ALLOW

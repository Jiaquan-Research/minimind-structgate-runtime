"""
Fusion Policy (AND)
===================
ALLOW only if all probe constraints are satisfied.
"""
from typing import Dict
from structgate.decision import GateAction

class AndFusionPolicy:
    def __init__(
        self,
        max_entropy: float,
        min_margin: float,
    ):
        self.max_entropy = max_entropy
        self.min_margin = min_margin

    def decide(self, obs: Dict) -> GateAction:
        # Expect obs to contain merged outputs from probes
        # Conservative defaults:
        # If entropy missing -> inf (fail)
        # If margin missing -> 0.0 (fail)
        entropy_val = obs.get("entropy", float("inf"))
        margin_val = obs.get("margin", 0.0)

        entropy_ok = entropy_val <= self.max_entropy
        margin_ok = margin_val >= self.min_margin

        if entropy_ok and margin_ok:
            return GateAction.ALLOW
        return GateAction.REFUSE
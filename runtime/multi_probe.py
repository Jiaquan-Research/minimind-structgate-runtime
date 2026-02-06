"""
MultiProbe Aggregator
=====================
Runs multiple probes and merges their observations.
"""
from typing import Any, Dict, List

class MultiProbe:
    def __init__(self, probes: List):
        self.probes = probes

    def observe(self, model_raw_output: Any) -> Dict:
        merged: Dict = {}
        for p in self.probes:
            out = p.observe(model_raw_output)
            merged.update(out)
        return merged
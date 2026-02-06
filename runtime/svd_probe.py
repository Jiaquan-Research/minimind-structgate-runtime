"""
SVD Probe (Stateful) - Phase 3.2 FINAL
====================================
Analyzes spectral properties of hidden-state trajectories.

ENGINEERING NOTE
----------------
This probe analyzes the accumulation buffer of hidden states passed
to the runtime.

- With static prompts: detects attractor / fixed-point anisotropy.
- With generation loops (future Phase 3.3): detects trajectory collapse.

INTERPRETATION
--------------
High sv_ratio => low-rank trajectory (anisotropy).
This is a NECESSARY but NOT SUFFICIENT condition for repetition or collapse.
"""

import torch
from typing import Any, List, Optional
from runtime.interface import ModelOutput


class SVDProbe:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer: List[torch.Tensor] = []

    def observe(self, model_raw_output: Any) -> ModelOutput:
        h_last = model_raw_output.get("last_hidden_state")

        if h_last is None:
            return {}

        # Update rolling buffer
        self.buffer.append(h_last.detach().clone())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Not ready → return None (IMPORTANT)
        if len(self.buffer) < self.window_size:
            return {
                "sv_ratio": None,
                "svd_ready": False,
            }

        # Stack into matrix: [window, hidden_dim]
        M = torch.stack(self.buffer)

        # Centering improves spectral interpretability
        M_centered = M - M.mean(dim=0)

        try:
            S = torch.linalg.svdvals(M_centered)
            total = S.sum() + 1e-9
            ratio = (S[0] / total).item()

            return {
                "sv_ratio": ratio,
                "svd_ready": True,
            }
        except Exception as e:
            print(f"[SVDProbe] numerical failure: {e}")
            return {
                "sv_ratio": None,
                "svd_ready": False,
            }

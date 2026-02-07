import torch
import numpy as np


class SVDProbe:
    """
    SVD-based structural probe.

    Supports two input modes:
    1. Dict-based trace (generation-time)
       { "last_hidden_state": Tensor }
    2. Tensor-based hidden state (training-time)
       Tensor[Dim] or Tensor[Batch, Seq, Dim]
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = []

    def _extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor to 1D feature vector [Dim].
        """
        if x.ndim == 1:
            return x
        if x.ndim == 2:
            # [Seq, Dim] → take last token
            return x[-1]
        if x.ndim == 3:
            # [Batch, Seq, Dim] → first sample, last token
            return x[0, -1]
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    def observe(self, model_output):
        """
        Observe model internal state and compute SVD ratio.

        Accepts:
        - dict with 'last_hidden_state'
        - torch.Tensor
        """
        # --- Normalize input ---
        if isinstance(model_output, dict):
            h = model_output.get("last_hidden_state", None)
            if h is None:
                return {}
            vec = self._extract_vector(h)

        elif torch.is_tensor(model_output):
            vec = self._extract_vector(model_output)

        else:
            return {}

        vec = vec.detach().cpu().float()

        # --- Sliding window ---
        self.buffer.append(vec.numpy())
        if len(self.buffer) < self.window_size:
            return {}

        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        X = np.stack(self.buffer)  # [W, Dim]

        # --- SVD ---
        try:
            _, s, _ = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return {}

        if s.sum() == 0:
            return {}

        sv_ratio = float(s[0] / s.sum())

        return {
            "sv_ratio": sv_ratio,
            "rank": int((s > 1e-6).sum())
        }

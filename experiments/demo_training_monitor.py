"""
StructGate Phase 3.4: Training-Time Telemetry (Concept Demonstration)
=====================================================================

Purpose:
  Demonstrate the theoretical foundation of the Early Warning System (EWS)
  for structural collapse using a minimal, self-contained Transformer.

Key Concepts:
  - White-box monitoring via Forward Hooks.
  - SVD-based spectral analysis of hidden states.
  - Separation of failure modes:
    1. Healthy Training
    2. Numeric Explosion (Loss divergence)
    3. Representation Collapse (Rank degradation)

Note:
  This script uses a synthetic 'MockGPT' model and synthetic data to isolate
  the monitoring logic from external dependencies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Assuming your package structure, otherwise adjust import
from runtime.svd_probe import SVDProbe


# ======================================================
# 1. Mock Dataset (Synthetic)
# ======================================================
class DummyDataset(Dataset):
    """
    Synthetic token dataset for pipeline validation.
    Contains no semantic information, used strictly to drive the training loop.
    """
    def __init__(self, vocab_size=1000, length=1000, seq_len=32):
        # Length is set to 1000 to ensure the loop runs long enough for the SVD buffer to fill
        self.data = torch.randint(0, vocab_size, (length, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ======================================================
# 2. Minimal Transformer (Self-Contained)
# ======================================================
class MockConfig:
    vocab_size = 1000
    dim = 512
    n_heads = 8
    n_layers = 4


class MockGPT(nn.Module):
    """
    A minimal Transformer stack implementation.
    Defined explicitly to demonstrate hook attachment on standard nn.Module structures.
    """
    def __init__(self, cfg: MockConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        # Using explicit ModuleList to guarantee hook accessibility
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.dim,
                nhead=cfg.n_heads,
                batch_first=True
            )
            for _ in range(cfg.n_layers)
        ])
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)
        return logits


# ======================================================
# 3. Telemetry Structures
# ======================================================
@dataclass
class HealthStatus:
    svd_ratio: Optional[float]  # None if the probe buffer is not yet full
    is_collapsed: bool
    raw_obs: Dict[str, Any]


class TrainingMonitor:
    def __init__(self, window_size=10, collapse_threshold=0.8):
        self.svd_probe = SVDProbe(window_size=window_size)
        self.collapse_threshold = collapse_threshold

    def analyze(self, hidden_state: torch.Tensor) -> HealthStatus:
        """
        Analyze the hidden state to detect structural anomalies.
        """
        # Sampling strategy: Observe the last token of the first sequence in the batch
        probe_vec = hidden_state[0, -1, :].detach().cpu()

        # Phase 3 Contract: Observe expects a Dictionary
        obs = self.svd_probe.observe({
            "last_hidden_state": probe_vec
        })

        svd_ratio = obs.get("sv_ratio")

        is_collapsed = False
        if svd_ratio is not None:
            is_collapsed = svd_ratio > self.collapse_threshold

        return HealthStatus(
            svd_ratio=svd_ratio,
            is_collapsed=is_collapsed,
            raw_obs=obs
        )


# ======================================================
# 4. Training Demo Runner
# ======================================================
def run_training_demo(mode: str):
    """
    Execute a training run under specific conditions to demonstrate detection capabilities.

    Args:
      mode: "healthy", "numeric_explosion", or "representation_collapse"
    """
    assert mode in {
        "healthy",
        "numeric_explosion",
        "representation_collapse"
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> [Phase 3.4] Training Monitor | Mode: {mode} | Device: {device}")

    # --- Model Setup ---
    model = MockGPT(MockConfig()).to(device)
    model.train()

    # --- Data Setup ---
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # --- Optimizer Setup ---
    if mode == "numeric_explosion":
        lr = 1.0  # Intentionally unstable learning rate to trigger loss divergence
    else:
        lr = 1e-4

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # --- Monitor Setup ---
    monitor = TrainingMonitor()
    captured: Dict[str, torch.Tensor] = {}

    # --- Forward Hook (Sidecar Pattern) ---
    def make_hook():
        step_counter = {"n": 0}

        def hook(module, inp, out):
            h = out.detach()

            # Simulate Representation Collapse (Artificial degradation of rank)
            if mode == "representation_collapse":
                step_counter["n"] += 1
                decay = min(1.0, step_counter["n"] / 20.0)
                h = h.clone()
                # Artificially squash dimensions to simulate rank collapse
                h[..., 1:] *= (1.0 - decay)

            captured["hidden"] = h

        return hook

    # Attach to the last transformer layer
    hook_handle = model.layers[-1].register_forward_hook(make_hook())
    print(">> [Hook] Successfully attached to TransformerEncoderLayer")

    # --- Logging Header ---
    print(f"{'Step':<6} | {'Loss':<10} | {'SVD':<8} | Status")
    print("-" * 50)

    # --- Training Loop ---
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            batch.view(-1)
        )

        loss.backward()
        optimizer.step()

        # Telemetry Analysis (Real-time)
        current_status = None
        if "hidden" in captured:
            current_status = monitor.analyze(captured["hidden"])

        # Logging & Alerting (Sampled every 5 steps)
        if step % 5 == 0 and current_status is not None:

            if current_status.svd_ratio is None:
                svd_str = "WAIT"
                icon = "⚪ BUFFERING"
            else:
                svd_str = f"{current_status.svd_ratio:.3f}"
                icon = "🔴 COLLAPSE" if current_status.is_collapsed else "🟢 HEALTHY"

            print(f"{step:<6} | {loss.item():<10.4f} | {svd_str:<8} | {icon}")

            # Circuit Breaker Logic
            if current_status.is_collapsed:
                print(f"\n[ALERT] Early Warning Triggered: Representation Collapse detected (SVD={svd_str}).")
                print("        Stopping training to prevent silent failure.")
                break

            if loss.item() > 50:
                print(f"\n[ALERT] Loss Explosion Triggered (Loss={loss.item():.2f}).")
                break

        if step >= 60: # Limit run duration for demo purposes
            break

    hook_handle.remove()


# ======================================================
# 5. Entry Point
# ======================================================
if __name__ == "__main__":
    run_training_demo("healthy")
    # run_training_demo("numeric_explosion")      # Uncomment to test loss divergence
    # run_training_demo("representation_collapse") # Uncomment to test SVD alarm
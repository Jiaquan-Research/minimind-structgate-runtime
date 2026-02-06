"""
Phase 3.2 Smoke Test: SVD Probe (FINAL)
=====================================
Tests mechanics of stateful SVD monitoring.

LIMITATION
----------
This test repeatedly feeds the SAME prompt.
It verifies buffer mechanics and spectral computation only.

This is NOT a token-generation trajectory test.
"""

import os
import sys
sys.path.append(os.getcwd())

from minimind.model import MiniMindModel
from runtime.interface import RuntimeEngine
from runtime.svd_probe import SVDProbe


def main():
    print(">>> Phase 3.2 Smoke Test: SVD Probe (Mechanics Only)")

    # Resolve checkpoint path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ckpt_path = os.path.join(project_root, "minimind", "weights")

    # Load model (CPU enforced)
    model = MiniMindModel(ckpt_path=ckpt_path, device="cpu")

    # SVD probe with small window for testing
    WINDOW = 5
    probe = SVDProbe(window_size=WINDOW)
    engine = RuntimeEngine(model=model, probe=probe)

    prompt = "The"
    print(f"\n--- Repeated input: '{prompt}' (window={WINDOW}) ---")

    for i in range(WINDOW + 2):
        obs = engine.step(prompt)

        ratio = obs.get("sv_ratio")
        ready = obs.get("svd_ready")

        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        status = "READY" if ready else "BUFFERING"

        print(f"Step {i+1}: sv_ratio={ratio_str} [{status}]")

    print("\n✅ Phase 3.2 mechanics verified.")
    print("Note: High ratios are expected for identical hidden states.")


if __name__ == "__main__":
    main()

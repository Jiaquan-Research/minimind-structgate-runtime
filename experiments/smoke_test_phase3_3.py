"""
Phase 3.3 Smoke Test: Generation Telemetry
=========================================
End-to-end test:
- Model: MiniMindModel (generate_with_trace)
- Engine: GenerationEngine
- Probes: Entropy + LayerDelta + SVD (stateful)
Observes per-token telemetry during real generation.
"""

import os, sys
sys.path.append(os.getcwd())

from minimind.model import MiniMindModel
from runtime.generation_engine import GenerationEngine
from runtime.multi_probe import MultiProbe

# Probes
from runtime.token_entropy_probe import TokenEntropyProbe
from runtime.internal_probes import LayerDeltaProbe, NormProbe
from runtime.svd_probe import SVDProbe


def main():
    print(">>> Phase 3.3 Smoke Test: Generation Telemetry")

    # --- Paths ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ckpt_path = os.path.join(project_root, "minimind", "weights")

    # --- Load Model ---
    try:
        model = MiniMindModel(ckpt_path=ckpt_path, device="cpu")
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    # --- Probes ---
    # SVD needs a window; small window for quick visibility
    svd_probe = SVDProbe(window_size=6)

    probe_suite = MultiProbe([
        TokenEntropyProbe(),     # entropy
        LayerDeltaProbe(),       # layer_delta
        NormProbe(),             # activation_energy (optional, logged)
        svd_probe                # sv_ratio (stateful)
    ])

    # --- Generation Engine ---
    engine = GenerationEngine(model=model, probe=probe_suite)

    # --- Test Prompt ---
    prompt = "Explain why the sky is blue in simple terms."
    max_tokens = 24

    # --- Run ---
    logs = engine.run(prompt, max_tokens=max_tokens, verbose=True)

    # --- Post-checks ---
    print("\n--- Post-run Sanity Checks ---")
    print(f"Total tokens generated: {len(logs)}")

    # Count SVD readiness
    ready_cnt = sum(1 for x in logs if x["metrics"].get("svd_ready"))
    print(f"SVD ready steps: {ready_cnt}")

    # Print last few entries compactly
    print("\n--- Tail (last 5 steps) ---")
    for entry in logs[-5:]:
        tok = entry["token"]
        m = entry["metrics"]
        sv = m.get("sv_ratio")
        dl = m.get("layer_delta")
        en = m.get("entropy")
        print(
            f"['{tok}'] | "
            f"SVD={sv if sv is not None else 'None'} | "
            f"Delta={dl:.4f} | "
            f"Ent={en:.4f}"
        )

    print("\n✅ Phase 3.3 Smoke Test COMPLETE.")


if __name__ == "__main__":
    main()

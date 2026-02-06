"""
Phase 3.1 Smoke Test: Internal Probes
=====================================
Verifies that white-box signals (Delta, Norm) are correctly computed.
"""
import os, sys

sys.path.append(os.getcwd())

from minimind.model import MiniMindModel
from runtime.interface import RuntimeEngine
from runtime.multi_probe import MultiProbe
# Import new probes
from runtime.internal_probes import LayerDeltaProbe, NormProbe
# Import old probes
from runtime.token_entropy_probe import TokenEntropyProbe
from runtime.margin_probe import MarginProbe


def main():
    print(">>> Phase 3.1 Smoke Test: Internal Probes")

    # 1. Load Model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ckpt_path = os.path.join(project_root, "minimind", "weights")

    try:
        model = MiniMindModel(ckpt_path=ckpt_path, device="cpu")
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    # 2. Setup All Probes
    probes_list = [
        TokenEntropyProbe(),
        MarginProbe(),
        LayerDeltaProbe(),
        NormProbe()
    ]
    probe = MultiProbe(probes_list)

    engine = RuntimeEngine(model=model, probe=probe)

    # 3. Test Inputs
    prompts = [
        "The capital of France is",  # Normal
        "sjfkal jklfsjdkl 32423",  # Nonsense
    ]

    for p in prompts:
        print(f"\n--- Input: '{p}' ---")
        obs = engine.step(p)

        # Print all metrics
        print(f"Entropy:       {obs.get('entropy'):.4f}")
        print(f"Margin:        {obs.get('margin'):.4f}")
        print(f"Layer Delta:   {obs.get('layer_delta'):.4f}")
        print(f"Activ Energy:  {obs.get('activation_energy'):.4f}")

        # Validation
        if obs.get('layer_delta') is not None and obs.get('activation_energy') is not None:
            print("✅ Internal metrics captured.")
        else:
            print("❌ Internal metrics MISSING.")


if __name__ == "__main__":
    main()
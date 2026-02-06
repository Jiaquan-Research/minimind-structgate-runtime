"""
Phase 2B Smoke Test
===================
Multi-probe + AND fusion policy.
"""
import os, sys

# Ensure root is on path
sys.path.append(os.getcwd())

from minimind.model import MiniMindModel
from runtime.interface import RuntimeEngine
from runtime.token_entropy_probe import TokenEntropyProbe
from runtime.margin_probe import MarginProbe
from runtime.multi_probe import MultiProbe
from structgate.decision import StructGate
from structgate.fusion_policy import AndFusionPolicy


def main():
    print(">>> Phase 2B Smoke Test Start")

    # Resolve absolute ckpt path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ckpt_path = os.path.join(project_root, "minimind", "weights")

    # Load Model (CPU)
    try:
        model = MiniMindModel(ckpt_path=ckpt_path, device="cpu")
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    # Probes: Entropy + Margin
    entropy_probe = TokenEntropyProbe()
    margin_probe = MarginProbe()
    # Fusion: MultiProbe acts as a single probe interface
    probe = MultiProbe([entropy_probe, margin_probe])

    engine = RuntimeEngine(model=model, probe=probe)

    # Policy (empirical calibration)
    # Entropy <= 6.5 AND Margin >= 0.02
    policy = AndFusionPolicy(
        max_entropy=6.5,
        min_margin=0.02,
    )
    gate = StructGate(policy=policy)

    cases = [
        ("Simple", "The capital of France is"),
        ("Nonsense", "sjfkal jklfsjdkl 32423 @@!!"),
    ]

    for name, prompt in cases:
        print(f"\n--- {name} ---")
        obs = engine.step(prompt)

        ent = obs.get('entropy')
        mar = obs.get('margin')
        print(f"Metrics: Entropy={ent:.4f}, Margin={mar:.6f}")

        action = gate.evaluate(obs)
        print(f"Gate:    {action}")


if __name__ == "__main__":
    main()
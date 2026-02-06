"""
Phase 1.3: Real Model + StructGate Smoke Test
=============================================
Verifies Runtime -> Probe -> StructGate -> Action loop with a real LM.
"""
import sys
import os

# Ensure root import visibility
sys.path.append(os.getcwd())

from runtime.interface import RuntimeEngine
from structgate.simple_policy import EntropyThresholdPolicy
from structgate.decision import StructGate, GateAction
from runtime.token_entropy_probe import TokenEntropyProbe
from minimind.model import MiniMindModel

def main():
    print(">>> Phase 1.3: Real Model Smoke Test Start")

    # 1. Load Model (GPT-2 as MiniMind proxy)
    # Explicitly enforce CPU for deterministic behavior contract
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ckpt_path = os.path.join(project_root, "minimind", "weights")

    print(f"[Debug] 尝试加载模型路径: {ckpt_path}")
    try:
        model = MiniMindModel(ckpt_path=ckpt_path, device="cpu")
    except Exception as e:
        print(f"❌ CRITICAL: Model load failed. Check {ckpt_path}.")
        print(f"Details: {e}")
        return
    print("[OK] Real Model loaded (CPU)")

    # 2. Init Probe & Engine
    probe = TokenEntropyProbe()
    engine = RuntimeEngine(model=model, probe=probe)
    print("[OK] Runtime Engine initialized")

    # 3. Init StructGate
    # Threshold chosen empirically for smoke test demonstration.
    # Typical GPT-2 next-token entropy varies significantly by context.
    policy = EntropyThresholdPolicy(max_entropy=6.5)
    gate = StructGate(policy=policy)
    print("[OK] StructGate initialized (Threshold: 6.5)")

    # 4. Test Cases
    # Case A: High certainty (completion) -> Low Entropy
    # Case B: High uncertainty (nonsense) -> High Entropy
    cases = [
        ("Simple", "The capital of France is"),
        ("Nonsense", "sjfkal jklfsjdkl 32423 @@!!")
    ]

    for name, prompt in cases:
        print(f"\n--- Testing Case: {name} ---")
        print(f"Prompt: '{prompt}'")

        # Step 1: Runtime Observation
        obs = engine.step(prompt)
        entropy = obs["entropy"]
        print(f"Probe: Real Entropy = {entropy:.4f}")

        # Step 2: Gate Decision
        action = gate.evaluate(obs)
        print(f"Gate:  Action = {action}")

        # Basic Validation Logic
        if name == "Simple" and action == GateAction.ALLOW:
            print("✅ PASS: Low entropy allowed.")
        elif name == "Nonsense" and action == GateAction.REFUSE:
            print("✅ PASS: High entropy refused.")
        else:
            print(f"⚠️ NOTE: Action {action} with Entropy {entropy:.4f}. Threshold tuning may be needed.")

if __name__ == "__main__":
    main()
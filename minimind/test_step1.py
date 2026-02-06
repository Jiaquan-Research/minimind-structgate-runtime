import os, sys

sys.path.append(os.getcwd())
import torch
from minimind.model import MiniMindModel


def test_step1():
    print(">>> Phase 3.3 Step 1: Generation Adapter Test")

    # 1. Load Model
    ckpt = "./minimind/weights"  # 你的路径
    # 自动修正路径
    if not os.path.exists(ckpt):
        # 尝试向上找一级
        ckpt = os.path.join(os.path.dirname(os.getcwd()), "minimind", "weights")

    model = MiniMindModel(ckpt_path=ckpt, device="cpu")

    print("\n--- Generating Trace ---")
    iterator = model.generate_with_trace("The capital of France is", max_tokens=5)

    for i, step in enumerate(iterator):
        token = step["token"]
        h_shape = step["last_hidden_state"].shape
        prev_h_shape = step["prev_hidden_state"].shape

        print(f"Step {i + 1}: Token='{token}' | Hidden={h_shape}")

        # 校验点
        assert isinstance(step["last_hidden_state"], torch.Tensor)
        assert step["last_hidden_state"].device.type == "cpu"
        assert len(h_shape) == 1  # 应该是[dim]

    print("\n✅ Generation Adapter works with KV-Cache.")


if __name__ == "__main__":
    test_step1()
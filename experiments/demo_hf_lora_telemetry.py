"""
StructGate Phase 3.5: Hugging Face & LoRA Integration Demo
==========================================================

Purpose:
  Demonstrate StructGate telemetry integration within a standard
  Industrial AI pipeline (Hugging Face Transformers + PEFT/LoRA).

Key Tech Stack:
  - Transformers (AutoModel, AutoTokenizer)
  - PEFT (LoRA for parameter-efficient fine-tuning)
  - StructGate (Runtime Telemetry Hook)

Scope:
  - Observation ONLY. No control policy applied.
  - Verifies that SVD probes work on real pre-trained LLMs (Qwen/Llama).
  - Implements Robust Layer Targeting via semantic search.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

# Import custom probe
from runtime.svd_probe import SVDProbe

# ------------------------------------------------
# 1. Config (Industrial Standard)
# ------------------------------------------------
# Using Qwen2.5-0.5B: Modern, fast, and lightweight for demonstration
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 30

print(f"\n>>> [StructGate-HF] Initializing Pipeline on {DEVICE}...")

# ------------------------------------------------
# 2. Load Pre-trained Model & Tokenizer
# ------------------------------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True
    ).to(DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ------------------------------------------------
# 3. Apply LoRA (Parameter-Efficient Fine-Tuning)
# ------------------------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print(f">> [PEFT] LoRA Adapter attached.")
# model.print_trainable_parameters() # Optional: Print parameter stats

# ------------------------------------------------
# 4. Pipeline Data (In-Memory Mock)
# ------------------------------------------------
# Generating synthetic data in-memory to avoid external file dependencies
raw_texts = [
    "StructGate is a runtime safety system for LLMs.",
    "Machine learning infrastructure requires observability.",
    "LoRA allows efficient fine-tuning of large models.",
    "SVD probes detect representation collapse early.",
    "Hugging Face transformers are the industry standard.",
] * 10

encoded_data = tokenizer(
    raw_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=32
)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataloader = DataLoader(SimpleDataset(encoded_data), batch_size=4, shuffle=True)

# ------------------------------------------------
# 5. Integrate StructGate (The "Magic")
# ------------------------------------------------
svd_probe = SVDProbe(window_size=5)
captured_state = {}

def infra_hook(module, input, output):
    # HF models typically return a tuple; we need the hidden states (first element)
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output
    captured_state["last_hidden"] = hidden.detach()

# --- Semantic Layer Hunter ---
# Instead of hardcoding paths (e.g., model.layers[-1]), which breaks with PEFT wrapping,
# we scan modules to find the last specific Transformer Block type.

def get_target_layer(model):
    target = None
    for name, module in model.named_modules():
        class_name = module.__class__.__name__

        # Explicit Whitelist of valid Transformer Block types
        # Covers: Qwen2, Llama, GPT-2, Mistral, etc.
        if "DecoderLayer" in class_name or "GPT2Block" in class_name:
            target = module # Keep updating until the last one is found

    return target

target_layer = get_target_layer(model)

if target_layer:
    handle = target_layer.register_forward_hook(infra_hook)
    print(f">> [Telemetry] Hook successfully attached to: {target_layer.__class__.__name__}")

    # Safety Check: Ensure we didn't attach to a non-block layer (e.g., RotaryEmbedding)
    if "Rotary" in target_layer.__class__.__name__:
        print("!! [CRITICAL] Attached to RotaryEmbedding. Layer logic failed.")
        exit(1)
else:
    print("!! [Error] Could not find any DecoderLayer/Block in the model.")
    exit(1)

# ------------------------------------------------
# 6. Training Loop (Standard HF Style)
# ------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("-" * 65)
print(f"{'Step':<6} | {'Loss':<10} | {'SVD Ratio':<10} | {'Infra Status'}")
print("-" * 65)

model.train()
step_count = 0

for batch in dataloader:
    if step_count >= MAX_STEPS: break

    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    outputs = model(**batch, labels=batch["input_ids"])
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # --- Telemetry Check ---
    status_icon = "⚪ WAIT"
    svd_val = 0.0

    if "last_hidden" in captured_state:
        # Sample the last token's hidden state
        vec = captured_state["last_hidden"][0, -1, :].cpu()

        obs = svd_probe.observe({"last_hidden_state": vec})
        svd_val = obs.get("sv_ratio", 0.0)

        if svd_val is None:
            svd_val = 0.0
            status_icon = "⚪ BUFF"
        elif svd_val > 0.8:
            status_icon = "🔴 COLLAPSE"
        else:
            status_icon = "🟢 HEALTHY"

    print(f"{step_count:<6} | {loss.item():<10.4f} | {svd_val:<10.3f} | {status_icon}")
    step_count += 1

# Cleanup
handle.remove()
print("\n>>> [StructGate-HF] Pipeline verification complete.")
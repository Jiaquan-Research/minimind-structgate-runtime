# Experiments — StructGate Phase 3 Demos

This directory contains **frozen, executable demonstrations** for
StructGate Phase 3 (runtime observation only).

These scripts are **not benchmarks**, **not training recipes**, and
**not production pipelines**.
They are **instrumentation probes** designed to show *how* telemetry is
captured and *what signals appear* under controlled conditions.

---

## Phase 3.4 — Training-Time Telemetry (Toy Model)

### `demo_training_monitor.py`

**Purpose**
- Demonstrate **runtime, white-box telemetry** during model training
- Show that **structural signals can emerge before loss divergence**
- Validate SVD-based probes under controlled synthetic failures

**What it demonstrates**
- Forward hooks attached to transformer layers
- Hidden-state capture during training
- SVD ratio as a structural indicator
- Separation of **observation** and **decision**:
  - The monitor only reports `HealthStatus`
  - Training loop decides whether to stop

**Modes**
- `healthy`: normal training
- `numeric_explosion`: loss diverges early
- `representation_collapse`: synthetic low-rank collapse with stable loss

**Scope constraints**
- CPU-only
- Toy-scale model
- Observation only (no intervention logic)

---

## Phase 3.5 — HuggingFace + LoRA Integration

### `demo_hf_lora_telemetry.py`

**Purpose**
- Prove StructGate telemetry compatibility with **industrial-standard tooling**
- Integrate runtime probes into a real HuggingFace training loop
- Demonstrate LoRA fine-tuning with telemetry hooks

**Tech stack**
- Hugging Face Transformers
- PEFT / LoRA
- Pretrained LLM (Qwen2.5-0.5B)
- PyTorch forward hooks

**What it demonstrates**
- Loading real pretrained LLMs
- Parameter-efficient fine-tuning (LoRA)
- Robust layer discovery under PEFT wrapping
- Runtime hidden-state telemetry on real models

**Expected behavior**
- During healthy LoRA fine-tuning:
  - SVD ratio remains in a moderate range (~0.5–0.7)
  - No collapse events are expected
- This is a **compatibility validation**, not a failure showcase

**Scope constraints**
- CPU-compatible demo
- Small batch, short run
- Observation only (no control policy)

---

## What This Directory Is NOT

- ❌ Not a benchmark suite
- ❌ Not a training optimization framework
- ❌ Not an alignment or safety solution
- ❌ Not claiming predictive guarantees

These demos exist to validate **observability**, not performance.

---

## Reproducibility Notes

- All scripts are single-file runnable
- No hidden configuration
- No external datasets required
- Deterministic behavior is preferred over realism

---

## Relationship to the Project

```

layer2-dynamics-probe  → inference-time diagnostics
demo_training_monitor → training-time telemetry (toy)
demo_hf_lora_telemetry → training-time telemetry (real HF stack)

```

Together, these define **StructGate Phase 3**:
> Measuring internal dynamics without acting on them.

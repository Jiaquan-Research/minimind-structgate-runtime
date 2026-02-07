# MiniMind-StructGate Runtime

**Phase 3: Runtime Telemetry for LLM Training and Inference**

MiniMind-StructGate is a **runtime observability and telemetry framework** for large language models (LLMs).  
It is designed as an **engineering-grade measurement instrument**, not an optimizer, policy learner, or safety solution.

The project focuses on **white-box, runtime-level probes** that monitor internal model dynamics during training and inference, under strict scope and falsifiability constraints.

---

## What This Project Is

- A **runtime telemetry system** for LLMs
- A **Phase 3 (observation-only)** artifact: no control, no optimization
- A collection of **internal probes** (e.g. SVD, entropy) attached via PyTorch hooks
- A bridge from **toy models → real Hugging Face training pipelines**
- A **reproducible, CPU-compatible** experimental framework

---

## What This Project Is NOT

See [`docs/not_this_project.md`](docs/not_this_project.md) for the full list.

In short, this project is **NOT**:

- ❌ A training algorithm or optimizer
- ❌ An alignment or safety solution
- ❌ A reinforcement learning agent
- ❌ A benchmark or leaderboard project
- ❌ A claim of general failure prediction
- ❌ A production control system
- ❌ A theory paper disguised as code

---

## Phase Overview

### Phase 3.4 — Training-Time Telemetry (Toy / Controlled)

**Goal:**  
Demonstrate that **structural signals** (e.g. representation collapse) can be observed **before loss divergence**, in controlled settings.

**Key properties:**
- White-box hooks into Transformer layers
- Sliding-window SVD probe on hidden states
- Explicit synthetic failure injection
- Clear separation: *monitor observes, loop decides*

**Entry point:**
```bash
python experiments/demo_training_monitor.py
````

---

### Phase 3.5 — Hugging Face + LoRA Integration

**Goal:**
Verify that the same telemetry probes are **compatible with industrial-standard pipelines**.

**What this phase demonstrates:**

* Hugging Face `transformers` integration
* PEFT / LoRA fine-tuning
* Runtime hooks on real pre-trained LLMs (e.g. Qwen2.5-0.5B)
* Observation-only behavior under healthy training

**Entry point:**

```bash
python experiments/demo_hf_lora_telemetry.py
```

> No collapse is expected in this demo.
> Absence of failure is the correct and honest result.

---

## Repository Structure

```
minimind-structgate-runtime/
│
├── README.md                 # This document
├── requirements.txt          # Minimal runtime dependencies
│
├── docs/
│   ├── scope.md              # Formal scope and phase boundaries
│   ├── not_this_project.md   # Explicit non-goals
│   └── safety_envelope.png   # Conceptual diagram
│
├── experiments/
│   ├── README.md                     # Experiment index and usage notes
│   ├── demo_training_monitor.py      # Phase 3.4 training telemetry demo
│   ├── demo_hf_lora_telemetry.py     # Phase 3.5 HF + LoRA integration demo
│   ├── demo_generation_telemetry.py
│   ├── demo_svd_mechanics.py
│   └── visualize_boundary.py
│
├── runtime/
│   ├── svd_probe.py           # Core SVD telemetry probe
│   ├── token_entropy_probe.py
│   ├── internal_probes.py
│   ├── multi_probe.py
│   ├── generation_engine.py
│   └── interface.py
│
├── structgate/
│   ├── decision.py            # Decision interface (no execution)
│   ├── fusion_policy.py
│   └── simple_policy.py
│
└── minimind/
    └── model.py               # Minimal internal toy model (no weights tracked)
```

> **Note:**
> Model weights are intentionally excluded from version control.
> All demos are reproducible without proprietary assets.

---

## Design Principles

1. **Observation ≠ Control**
   Telemetry modules never stop training or alter gradients.

2. **Human-in-the-Loop Verification**
   All probes are heuristic, explicitly labeled, and auditable.

3. **Falsifiability First**
   Claims are limited to what the demos actually demonstrate.

4. **Compatibility over Performance**
   The goal is pipeline integration, not SOTA metrics.

---

## Development Protocol (AI-Assisted Engineering)

This project follows a strict **human-in-the-loop verification workflow**.

While LLMs were used as implementation assistants, all code is subject to:

* Explicit scope constraints
* Cross-model adversarial review
* Local execution audits
* Manual ownership of all design decisions

No code is merged unless it runs, aligns with documented assumptions, and respects the defined falsification boundaries.

---

## Relationship to Other Projects

This repository fits into a larger, coherent research–engineering arc:

```
Info-Flow-Dynamics
        ↓
Vacuum-X × Snake-SHM
        ↓
MiniMind-StructGate Runtime (this repo)
```

Each layer builds on the previous one without retroactive justification or scope creep.

---

## Status

* **Phase 3.4:** Frozen
* **Phase 3.5:** Frozen
* **Future phases:** Out of scope for this repository

---

## License

MIT


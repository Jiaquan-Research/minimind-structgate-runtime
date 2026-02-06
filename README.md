
# MiniMind × StructGate Runtime

**White-Box Runtime Telemetry for LLM Generation**

> A runtime-level, white-box monitoring instrument for large language models.
> Focused on *observability*, not optimization or autonomy.

---

## 1. What This Project Is

This project is an **engineering instrument**, not an agent framework.

It provides **runtime telemetry** for large language model inference and generation, exposing both **output-level uncertainty** and **internal representational dynamics** in real time.

The system is designed to answer one core question:

> *“What is the model’s cognitive / structural state **while it is generating**, not just after it finishes?”*

---

## 2. What This Project Is NOT (Explicit Boundaries)

To avoid category errors, this project explicitly **does NOT** aim to be:

* ❌ An autonomous agent
* ❌ An alignment solution
* ❌ A safety guarantee
* ❌ A prompt-engineering framework
* ❌ A model optimizer or trainer
* ❌ A controller that modifies generation behavior

**Phase 3 is strictly observational.**
No interventions. No steering. No feedback loops.

---

## 3. System Overview

At a high level, the runtime stack is:

```
Model Adapter
   ↓
Runtime / Generation Engine
   ↓
Probe Suite (Telemetry)
   ↓
StructGate (Decision Interface)
```

Key design principle:

> **Separate observation from action.**
> Telemetry first. Policy later.

---
graph TD
    User[User Prompt] --> Adapter[Generation Adapter]
    Adapter --> Engine[Generation Engine]
    
    subgraph "Phase 3: Telemetry Loop"
        Engine -- Token & States --> Probes[Probe Suite]
        Probes -- Analyze --> Metrics
        
        subgraph "Probe Layers"
            Metrics --> Output[Output: Entropy/Margin]
            Metrics --> Internal[Internal: Delta/Norm]
            Metrics --> Temporal[Trajectory: SVD Ratio]
        end
    end
    
    Metrics -- Telemetry Data --> Gate[StructGate Interface]
    Gate -- Policy Check --> Decision{Decision}
    Decision -- ALLOW --> Log[Log & Continue]
    Decision -- REFUSE --> Signal[Signal Flag (No Action Yet)]
    
    style User fill:#f9f,stroke:#333
    style Gate fill:#bbf,stroke:#333
    style Probes fill:#dfd,stroke:#333

## 4. Phase Roadmap (Current Status)

| Phase   | Name                        | Status       |
| ------- | --------------------------- | ------------ |
| Phase 1 | Runtime plumbing            | ✅ Complete   |
| Phase 2 | Output-level probes         | ✅ Complete   |
| Phase 3 | White-box runtime telemetry | ✅ **Frozen** |
| Phase 4 | Alarm / action interface    | ⏸ Planned    |
| Phase 5 | Meta-control / outer loop   | ⏳ Future     |

This repository currently **freezes at Phase 3**.

---

## 5. Phase 3: White-Box Runtime Telemetry (Core Contribution)

Phase 3 upgrades the system from **black-box output inspection** to **white-box runtime observability**, including **token-by-token internal signals during generation**.

### 5.1 Probe Taxonomy

Phase 3 probes are deliberately structured into three layers.

---

### A. Output Probes (Surface Signals)

Operate purely on model logits.

* **Entropy**

  * Measures epistemic uncertainty of next-token distribution
  * High entropy ⇒ model confusion / underspecification

* **Margin**

  * Difference between top-1 and top-2 probabilities
  * Low margin ⇒ fragile confidence

These are *necessary but shallow* indicators.

---

### B. Instant Internal Probes (Stateless, White-Box)

Operate on hidden states of the final transformer layers.

* **Layer Delta**

  * `1 − cosine_similarity(last_layer, prev_layer)`
  * High delta ⇒ internal reasoning discontinuity

* **Activation Energy (Norm)**

  * L2 norm of final hidden state
  * Abnormally high or low norms often correlate with OOD or degraded semantic activation

These probes expose **structural stress** inside the network.

---

### C. Trajectory Probes (Stateful, Temporal)

Operate over **time**, not single tokens.

* **SVD Probe**

  * Maintains a sliding window of hidden states
  * Computes singular value spectrum of the trajectory
  * Key metric: **top-1 singular value ratio**

Interpretation:

* High ratio ⇒ low-rank trajectory
* Indicates representation collapse, repetition attractors, or degenerate dynamics

⚠️ Engineering note:

> SVD ratio is a **necessary but not sufficient** condition for collapse.
> Phase 3 makes this explicit and avoids semantic over-claims.

---

## 6. Generation Telemetry (Phase 3.3)

Phase 3.3 introduces a **generation-aware runtime engine**.

### GenerationEngine

* Pull-based token generation
* Observes **every generated token**
* Feeds continuous telemetry into the probe suite
* Acts as a **flight data recorder**, not a controller

Logged per token:

```json
{
  "token": "<str>",
  "metrics": {
    "entropy": "<float>",
    "margin": "<float>",
    "layer_delta": "<float>",
    "activation_energy": "<float>",
    "sv_ratio": "<float>"
  }
}
```

This enables **time-series analysis** of internal model dynamics.

---

## 7. StructGate: Decision Interface (Phase 3 Scope)

StructGate in this phase acts only as a **decision interface**, not an executor.

* Receives merged telemetry
* Applies explicit, inspectable policies
* Emits symbolic actions (e.g. `ALLOW`, `REFUSE`)

Important constraint:

> **StructGate does not modify generation in Phase 3.**

It exists to formalize the boundary between **measurement** and **action**, preparing for Phase 4 without crossing it.

---

## 8. Why This Matters (Engineering Perspective)

Most LLM tooling evaluates models **after the fact**.

This system enables:

* Observing **instability before failure**
* Distinguishing:

  * high uncertainty vs
  * structural inconsistency vs
  * trajectory collapse
* Treating LLMs as **dynamic systems**, not static text generators

The mental model is closer to:

> **Industrial instrumentation**
> (flight recorders, engine telemetry, alarm sensors)

—not autonomous agents.

---

## 9. Design Philosophy

* Deterministic execution (explicit device control)
* Explicit data contracts
* Clear phase boundaries
* No hidden autonomy
* No semantic over-claims

The project prioritizes **engineering honesty** over flashy demos.

---

## 10. What Comes Next (Not Implemented Here)

Future phases are intentionally **out of scope** for this repository version:

* Phase 4: Alarm → Action mapping (modular, system-specific)
* Phase 5: Meta-control / goal evolution (outer loop)

These are deferred to avoid collapsing observation and control into an unsafe abstraction.

---

## 11. Intended Audience

* Systems engineers
* Control / safety researchers
* Runtime infrastructure developers
* Engineers evaluating LLM behavior under uncertainty

This is **not** a beginner tutorial and **not** a consumer AI product.

---

## 12. Status

**Phase 3 complete. Frozen for review, reuse, and integration.**


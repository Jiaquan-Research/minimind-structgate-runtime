# Project Scope

This repository implements a **runtime constraint and refusal layer**
(`StructGate`) applied to a **small language model**
(`MiniMind`) under conditions of **information insufficiency**.

The project is an **engineering experiment**, not a model improvement effort.

---

## In Scope

### 1. Runtime-Level Control Only
- StructGate operates **at inference time**
- No gradient updates
- No retraining
- No fine-tuning
- No reward shaping

The base model remains unchanged.

---

### 2. Information Insufficiency Detection
The system explicitly detects states where:
- Input information is incomplete, underspecified, or ambiguous
- The model's output distribution exhibits high uncertainty
- Confident generation would constitute hallucination risk

This detection is implemented via **observable signals** (e.g. entropy, logit dispersion).

---

### 3. Admissible Control Outputs
Under detected insufficiency, the controller may output:
- REFUSE
- NOOP
- DELAY / REQUEST MORE INFORMATION

These are treated as **valid control actions**, not failures.

---

### 4. Deterministic, Reproducible Experiments
- CPU-only
- Deterministic seeds
- Fixed prompts per experiment
- Identical model weights across all runs

---

### 5. Comparative Evaluation
Experiments are limited to:
- Baseline MiniMind (unconstrained)
- StructGate-protected MiniMind

Metrics focus on **failure mode reshaping**, not performance maximization.

---

## Core Claim (Operational)

> Under information insufficiency, introducing a runtime constraint layer
> increases the rate of *safe failure* (refusal or inaction)
> relative to unconstrained generation.

This claim is **empirical and falsifiable** within this repository.

---

## Out of Scope

Anything not explicitly listed above is out of scope.
See `not_this_project.md` for explicit exclusions.

"""
Runtime Interface Contract
==========================

This module defines the *minimal runtime contract* between:

    MiniMind (or any language model)
        →
    Runtime Probes
        →
    StructGate

Design principles:
- Model-agnostic: runtime must not depend on MiniMind internals
- Deterministic-friendly: no hidden state mutation
- Probe-first: runtime exposes *observables*, not decisions

This file intentionally contains NO implementation.
"""

from typing import Any, Dict, List, Protocol, TypedDict


class ModelOutput(TypedDict):
    """
    Canonical model output observable.

    This is the *only* structure Runtime is allowed to expose
    upward to StructGate.
    """
    logits: Any
    token_probs: List[float]
    entropy: float


class RuntimeProbe(Protocol):
    """
    Runtime probe interface.

    A probe extracts *diagnostic signals* from model execution.
    """

    def observe(self, model_raw_output: Any) -> ModelOutput:
        """
        Extract observables from raw model output.

        Parameters
        ----------
        model_raw_output:
            Raw forward output from the language model.

        Returns
        -------
        ModelOutput
            Structured, normalized observables for control.
        """
        ...


class LanguageModel(Protocol):
    """
    Minimal language model interface required by runtime.

    The runtime does NOT care how the model is trained or implemented.
    """

    def forward(self, prompt: str) -> Any:
        """
        Run a forward pass.

        Parameters
        ----------
        prompt : str
            Input text prompt.

        Returns
        -------
        Any
            Raw model output (implementation-defined).
        """
        ...


class RuntimeEngine:
    """
    Runtime orchestration layer (Implementation).
    """

    def __init__(
            self,
            model: LanguageModel,
            probe: RuntimeProbe,
    ):
        self.model = model
        self.probe = probe

    def step(self, prompt: str) -> ModelOutput:
        # 1. Raw Model Execution
        raw_output = self.model.forward(prompt)

        # 2. Probe Observation
        observables = self.probe.observe(raw_output)

        return observables

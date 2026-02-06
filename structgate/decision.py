"""
StructGate Decision Contract
============================

This module defines the *control-layer decision interface*.

StructGate is a:
- Read-only observer of runtime signals
- Deterministic decision module
- No learning, no adaptation, no memory (unless explicitly designed)

This file intentionally contains NO implementation.
"""

from enum import Enum
from typing import Dict, Protocol

from runtime.interface import ModelOutput


class GateAction(str, Enum):
    """
    Allowed control actions.

    These are *semantic* actions, not model tokens.
    """
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"
    NOOP = "NOOP"


class StructGatePolicy(Protocol):
    """
    StructGate policy interface.

    A policy maps runtime observables to admissible actions.
    """

    def decide(self, obs: ModelOutput) -> GateAction:
        """
        Make a control decision based on observables.

        Parameters
        ----------
        obs : ModelOutput
            Runtime observables (entropy, logits, etc.)

        Returns
        -------
        GateAction
            Control decision.
        """
        ...


class StructGate:
    """
    StructGate control wrapper (Implementation).
    """

    def __init__(self, policy: StructGatePolicy):
        self.policy = policy

    def evaluate(self, obs: ModelOutput) -> GateAction:
        # 1. Policy Decision
        action = self.policy.decide(obs)

        # (Future: Logging / Telemetry hooks go here)

        return action

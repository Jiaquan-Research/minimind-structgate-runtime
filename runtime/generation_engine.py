"""
Generation Runtime Engine (Phase 3.3)
=====================================
Orchestrates the generation loop and feeds continuous telemetry to probes.
Acts as a passive "Flight Data Recorder" - no interventions yet.
"""
from typing import List, Dict, Any


class GenerationEngine:
    def __init__(self, model, probe):
        """
        :param model: Must implement generate_with_trace()
        :param probe: MultiProbe or single Probe
        """
        self.model = model
        self.probe = probe
        # Stores the full history of the generation session
        self.logs: List[Dict] = []

    def run(self, prompt: str, max_tokens: int = 32, verbose: bool = True) -> List[Dict]:
        """
        Executes the generation loop.

        Returns:
            List of log entries (one per token), containing:
            - token: The generated token string
            - metrics: The probe outputs (entropy, sv_ratio, etc.)
        """
        print(f"--- [GenEngine] Start: '{prompt}' ---")

        # Clear previous logs
        self.logs = []

        # The Pull-Pattern: We ask the model for the next step
        iterator = self.model.generate_with_trace(prompt, max_tokens=max_tokens)

        for step_data in iterator:
            # step_data contains: token, logits, last_hidden_state, prev_hidden_state

            # 1. Probe Observation
            # The probe sees the internal state *before* we decide what to do
            metrics = self.probe.observe(step_data)

            # 2. Logging
            token_str = step_data["token"]
            log_entry = {
                "token": token_str,
                "metrics": metrics
            }
            self.logs.append(log_entry)

            # 3. Real-time Dashboard (Optional, for debugging)
            if verbose:
                self._print_dashboard(token_str, metrics)

        print("--- [GenEngine] Complete ---")
        return self.logs

    def _print_dashboard(self, token: str, metrics: Dict):
        """Helper to print a compact status line."""
        # SVD Ratio (Key Indicator)
        sv_ratio = metrics.get("sv_ratio")
        sv_str = f"{sv_ratio:.4f}" if sv_ratio is not None else "WAIT"

        # Layer Delta (Structural Stability)
        delta = metrics.get("layer_delta")
        delta_str = f"{delta:.4f}" if delta is not None else "----"

        # Entropy (Uncertainty)
        ent = metrics.get("entropy")
        ent_str = f"{ent:.4f}" if ent is not None else "----"

        print(f"['{token}'] \t| SVD: {sv_str} | Delta: {delta_str} | Ent: {ent_str}")
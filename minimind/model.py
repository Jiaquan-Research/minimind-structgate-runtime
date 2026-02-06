# minimind/model.py

import torch
from typing import Any, Dict, Iterator
from transformers import AutoTokenizer, AutoModelForCausalLM


class MiniMindModel:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"[MiniMind] Loading from {ckpt_path} on {self.device} (White-Box)...")

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        # 明确打开白盒输出
        self.model.config.output_hidden_states = True
        self.model.config.use_cache = True

    def forward(self, prompt: str) -> Dict[str, Any]:
        """
        Phase 3.1 / 3.2 使用的单步接口（保持不变）
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        next_token_logits = outputs.logits[0, -1, :].cpu().tolist()
        last_hidden = outputs.hidden_states[-1][0, -1, :].detach().cpu()
        prev_hidden = outputs.hidden_states[-2][0, -1, :].detach().cpu()

        return {
            "logits": next_token_logits,
            "last_hidden_state": last_hidden,
            "prev_hidden_state": prev_hidden,
        }

    # ===============================
    # Phase 3.3 NEW API
    # ===============================
    def generate_with_trace(
        self,
        prompt: str,
        max_tokens: int = 32,
    ) -> Iterator[Dict[str, Any]]:
        """
        Phase 3.3:
        Token-by-token generation with internal state tracing.

        Yields per-token:
        {
            "token": str,
            "logits": List[float],
            "last_hidden_state": Tensor [dim],
            "prev_hidden_state": Tensor [dim],
        }
        """

        # 1. tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        past_key_values = None

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # logits: [1, seq_len, vocab]
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # greedy decode（Phase 3.3 不引入 sampling 复杂性）
            next_token_id = torch.argmax(probs, dim=-1)

            token_str = self.tokenizer.decode(
                next_token_id[0],
                skip_special_tokens=True
            )

            # hidden states
            last_hidden = outputs.hidden_states[-1][0, -1, :].detach().cpu()
            prev_hidden = outputs.hidden_states[-2][0, -1, :].detach().cpu()

            yield {
                "token": token_str,
                "logits": logits[0].detach().cpu().tolist(),
                "last_hidden_state": last_hidden,
                "prev_hidden_state": prev_hidden,
            }

            # prepare next step
            past_key_values = outputs.past_key_values
            input_ids = next_token_id.unsqueeze(0)

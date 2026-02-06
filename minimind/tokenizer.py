"""
MiniMind Tokenizer
==================
Minimal tokenizer adapter.
"""


class MiniMindTokenizer:
    def __init__(self):
        # You can hardcode vocab or load from file
        pass

    def encode(self, text: str):
        """
        Encode text into list of integer token ids.
        """
        # Minimal BPE-style or whitespace-based mock
        # Replace with actual tokenizer logic if available
        tokens = text.strip().split()
        return [hash(w) % 1000 for w in tokens]  # example stub

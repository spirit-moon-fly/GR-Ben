import gc
from typing import List

import torch


class PRMAdapter:
    """Base class for all PRM adapters."""

    adapter_type: str = "base"

    def __init__(self, model_key: str, model_path: str, device: str):
        self.model_key = model_key
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        raise NotImplementedError

    def unload(self):
        del self.model, self.tokenizer
        self.model, self.tokenizer = None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_steps(self, query: str, steps: List[str]) -> List[float]:
        """Return a score in [0,1] per step. Higher = more likely correct."""
        raise NotImplementedError

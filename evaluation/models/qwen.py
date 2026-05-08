from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from .base import PRMAdapter


class QwenAdapter(PRMAdapter):
    """
    Adapter for Qwen probabilistic PRM models.

    Supported models:
      - Qwen2.5-Math-PRM-7B   (https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)
      - Qwen2.5-Math-PRM-72B  (https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B)
      - Qwen2.5-Math-7B-PRM800K

    Each step is appended with a special <extra_0> separator token.
    The model outputs a probability distribution at each separator position;
    index 1 corresponds to the "correct" class.

    Threshold strategy: fixed (0.5).
    """

    adapter_type = "qwen"

    def load(self):
        tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
        ).eval()
        self.tokenizer = tok
        self.model = model
        self.step_sep_id = tok.encode("<extra_0>")[0]

    def score_steps(self, query: str, steps: List[str]) -> List[float]:
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
        conversation_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(conversation_str, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_masks = input_ids == self.step_sep_id
        step_logits = logits[0][token_masks[0]]
        probs = torch.softmax(step_logits, dim=-1)[:, 1].cpu().float().tolist()
        return probs

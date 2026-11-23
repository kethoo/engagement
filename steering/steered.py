"""Frozen LLaMA split at a layer, so a steering vector can be added there.

``prefix(ids, key)`` runs the embeddings and decoder blocks ``0..key`` and returns their output
(= ``hidden_states[key + 1]``; repeng's layer key ``key``). It does not depend on the vector,
so it is computed once per article. ``suffix(h, key, end, vec)`` adds ``vec`` to every token
of that output — what repeng's ``ControlModel`` does at block ``key`` — and runs blocks
``key+1 .. end-1``, returning ``hidden_states[end]`` (the baseline head reads ``end = 16``,
mean-pooled). With ``vec = 0``, ``suffix(prefix(ids))`` is the unsteered forward pass.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask


class SteeredLlama:
    def __init__(self, model_path: str, dtype=torch.bfloat16, device: str = 'cuda'):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype, device_map=device)
        self.model.eval().requires_grad_(False)
        self.device = self.model.device

    def encode(self, prompt: str) -> torch.Tensor:
        # prompts from engagement.features.build_prompt already carry BOS
        return self.tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)

    def _ctx(self, x: torch.Tensor):
        pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        mask = create_causal_mask(config=self.model.config, inputs_embeds=x, attention_mask=None,
                                  past_key_values=None, position_ids=pos)
        return dict(attention_mask=mask, position_ids=pos, position_embeddings=self.model.rotary_emb(x, position_ids=pos))

    def _run(self, x, layers):
        ctx = self._ctx(x)
        for layer in layers:
            x = layer(x, **ctx)
            x = x[0] if isinstance(x, tuple) else x
        return x

    @torch.no_grad()
    def prefix(self, ids: torch.Tensor, key: int) -> torch.Tensor:
        """[1, T] token ids -> [1, T, d] output of block ``key``."""
        return self._run(self.model.embed_tokens(ids), self.model.layers[: key + 1])

    def suffix(self, h: torch.Tensor, key: int, end: int, vec: torch.Tensor | None = None) -> torch.Tensor:
        """[B, T, d] block-``key`` output (+ ``vec`` [B, d]) -> [B, T, d] ``hidden_states[end]``."""
        if vec is not None:
            h = h + vec[:, None, :].to(h.dtype)
        return self._run(h, self.model.layers[key + 1: end])

"""Hidden states of a frozen decoder (LLaMA) for each article, computed once and cached.

The backbone is frozen, so an article's representation never changes during head
training: one forward pass per article (~100 of them) replaces one per essay per
epoch. Every layer is stored with two poolings, so layer/pooling selection is a
head-training choice, not a re-extraction.

Cache layout (``torch.save`` dict):
    article_keys  list[str]                 row order
    mean, last    float16 [N, L+1, d]       hidden_states[0..L] (0 = embeddings)
    n_tokens      int [N]                   prompt length after truncation
    truncated     bool [N]
    meta          dict                      model, prompt, max_article_tokens
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def cache_path(cfg: dict) -> Path:
    f = cfg['features']
    sig = json.dumps({k: f[k] for k in ('model', 'prompt', 'max_article_tokens', 'use_chat_template')}, sort_keys=True)
    slug = f['model'].replace('/', '__')
    return Path(f['cache_dir']) / f'{slug}_{hashlib.sha1(sig.encode()).hexdigest()[:8]}.pt'


def _truncate(tok, text: str, max_tokens: int) -> tuple[str, bool]:
    ids = tok(text, add_special_tokens=False)['input_ids']
    if len(ids) <= max_tokens:
        return text, False
    return tok.decode(ids[:max_tokens]), True


def build_prompt(tok, title: str, text: str, cfg: dict) -> tuple[str, bool]:
    f = cfg['features']
    text, truncated = _truncate(tok, text, f['max_article_tokens'])
    content = f['prompt'].format(title=title, text=text).strip()
    if f.get('use_chat_template') and tok.chat_template:
        prompt = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    else:
        prompt = (tok.bos_token or '') + content
    return prompt, truncated


@torch.no_grad()
def extract(cfg: dict, articles: pd.DataFrame) -> Path:
    f = cfg['features']
    out = cache_path(cfg)
    if out.exists():
        cached = set(torch.load(out, weights_only=False)['article_keys'])
        if set(articles.article_key) <= cached:
            print(f'features cached: {out}')
            return out
        print(f'cache {out} lacks {len(set(articles.article_key) - cached)} articles — re-extracting')

    dtype = getattr(torch, f.get('dtype', 'bfloat16'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(f['model'])
    model = AutoModel.from_pretrained(f['model'], dtype=dtype, device_map=device if device == 'cuda' else None)
    model.eval().requires_grad_(False)

    means, lasts, n_tokens, truncated = [], [], [], []
    for i, row in enumerate(articles.itertuples()):
        prompt, trunc = build_prompt(tok, row.title, row.text, cfg)
        # the template already carries BOS — do not add a second one
        enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # (L+1) x [1, T, d]
        means.append(torch.stack([h[0].float().mean(0) for h in hs]).half().cpu())  # [L+1, d]
        lasts.append(torch.stack([h[0, -1].float() for h in hs]).half().cpu())
        n_tokens.append(enc['input_ids'].shape[1])
        truncated.append(trunc)
        if i % 10 == 0:
            print(f'[{i + 1}/{len(articles)}] {row.article_key} tokens={n_tokens[-1]} truncated={trunc}')

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'article_keys': list(articles.article_key),
        'mean': torch.stack(means),
        'last': torch.stack(lasts),
        'n_tokens': torch.tensor(n_tokens),
        'truncated': torch.tensor(truncated),
        'meta': {k: f[k] for k in ('model', 'prompt', 'max_article_tokens', 'use_chat_template')},
    }, out)
    print(f'saved {out}  shape={tuple(torch.stack(means).shape)}  truncated={sum(truncated)}/{len(truncated)}')
    return out


class FeatureStore:
    """Row lookup into the cache: ``X(keys, layer, pooling) -> np.ndarray [len(keys), d]``."""

    def __init__(self, path: str | Path):
        c = torch.load(path, weights_only=False)
        self.keys = {k: i for i, k in enumerate(c['article_keys'])}
        self.data = {'mean': c['mean'].float().numpy(), 'last': c['last'].float().numpy()}
        self.n_layers = c['mean'].shape[1] - 1  # hidden_states[0] is the embedding layer
        self.meta = c['meta']

    def X(self, keys, layer: int, pooling: str):
        layer = self.n_layers + 1 + layer if layer < 0 else layer
        return self.data[pooling][[self.keys[k] for k in keys], layer]

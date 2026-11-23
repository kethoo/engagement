"""Compute Big Five steering directions with repeng, exactly as Steer2Adapt's ``train.py``.

The procedure is a line-by-line copy of their ``train.py`` + ``utils.make_dataset``, so the
result can be checked against their published vectors (``compare_vectors.py``):

1. suffixes = ``all_truncated_outputs.json``, each expanded into its token prefixes;
2. for every suffix and every (positive, negative) persona pair (3 per trait):
       positive = "{user_tag} Act as if you're {positive persona}. {asst_tag} {suffix}"
       negative = the same with the negative persona;
3. ``repeng.ControlVector.train(..., method="pca_center", batch_size=32)``: last-token hidden
   state of every string at every layer; per layer, PC1 of the pair-centred states, signed so
   that positives project higher.

On top of their output this also saves, per layer, the mean positive and mean negative hidden
state (taken before ``pca_center`` centres the states in place), so the difference-of-means
direction is available without a second pass.

Layer keys are repeng's: key ``k`` = output of decoder block ``k`` = ``hidden_states[k + 1]``.
The baseline head's "layer 16" (``hidden_states[16]``) is therefore key 15, and Steer2Adapt's
Llama control layer ``-10`` is key 22.

    python -m steering.compute_vectors --trait agreeableness
    python -m steering.compute_vectors --trait all
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from repeng import ControlModel, ControlVector, DatasetEntry
import repeng.extract as rx
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import TRAITS

S2A_DIR = Path(__file__).parent / 'steer2adapt'
DTYPES = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}


def load_configs(trait: str, model_key: str | None, s2a_dir: Path = S2A_DIR):
    vc = yaml.safe_load(open(s2a_dir / f'{trait}.yaml'))
    key = model_key or vc['model']
    return vc, key, yaml.safe_load(open(s2a_dir / 'models.yaml'))[key]


def make_suffixes(vc: dict, tok, s2a_dir: Path = S2A_DIR) -> list[str]:
    """``train.py`` lines 46-62."""
    raw = json.load(open(s2a_dir / Path(vc['suffix_data']).name))
    if not vc.get('truncate_suffixes', False):
        return raw
    off = vc.get('truncate_offset', 0)
    return [
        tok.convert_tokens_to_string(tokens[:i])
        for tokens in (tok.tokenize(s) for s in raw)
        for i in range(1, len(tokens) + off)
    ]


def make_dataset(vc: dict, mc: dict, suffixes: list[str]) -> list[DatasetEntry]:
    """``utils.make_dataset``: one entry per (suffix, persona pair)."""
    out = []
    for suffix in suffixes:
        for pos, neg in zip(vc['positive_personas'], vc['negative_personas']):
            p, n = vc['template'].format(persona=pos), vc['template'].format(persona=neg)
            out.append(DatasetEntry(
                positive=f"{mc['user_tag']} {p} {mc['asst_tag']} {suffix}",
                negative=f"{mc['user_tag']} {n} {mc['asst_tag']} {suffix}",
            ))
    return out


class _MeanCapture:
    """Wraps ``repeng.extract.batched_get_hiddens`` to keep per-layer positive/negative means.

    repeng's ``pca_center`` subtracts the pair centre from the hidden states in place, so the
    means are taken here, before ``read_representations`` sees the array.
    """

    def __init__(self):
        self.orig, self.stats = rx.batched_get_hiddens, {}

    def __enter__(self):
        def wrapped(*args, **kwargs):
            hs = self.orig(*args, **kwargs)
            for layer, h in hs.items():  # rows are [pos, neg, pos, neg, ...]
                self.stats[layer] = (h[::2].mean(0), h[1::2].mean(0))
            return hs
        rx.batched_get_hiddens = wrapped
        return self

    def __exit__(self, *exc):
        rx.batched_get_hiddens = self.orig


def compute(trait: str, model, tok, model_key: str, mc: dict, out_dir: Path) -> dict:
    vc, _, _ = load_configs(trait, model_key)
    suffixes = make_suffixes(vc, tok)
    dataset = make_dataset(vc, mc, suffixes)
    print(f'[{trait}] {len(suffixes)} suffixes x {len(vc["positive_personas"])} persona pairs = {len(dataset)} pairs')
    print(f'[{trait}] example positive: {dataset[len(dataset) // 2].positive!r}')

    t0 = time.time()
    with _MeanCapture() as cap:
        cv = ControlVector.train(model, tok, dataset, method=vc['method'], batch_size=vc['batch_size'])
    secs = time.time() - t0

    stem = out_dir / f'{trait}_{model_key}'
    torch.save(cv, f'{stem}.pt')  # same object/format as Steer2Adapt's vectors/*.pt
    layers = sorted(cv.directions)
    pos = np.stack([cap.stats[l][0] for l in layers])
    neg = np.stack([cap.stats[l][1] for l in layers])
    np.savez(
        f'{stem}.npz',
        layers=np.array(layers),
        pca_center=np.stack([cv.directions[l] for l in layers]),  # unit norm, repeng's output
        diffmean=pos - neg,
        pos_mean=pos,
        neg_mean=neg,
    )
    meta = {
        'trait': trait, 'model_key': model_key, 'model': mc['model_name'],
        'model_path': mc.get('model_path'), 'dtype': mc['dtype'],
        'method': vc['method'], 'batch_size': vc['batch_size'], 'template': vc['template'],
        'positive_personas': vc['positive_personas'], 'negative_personas': vc['negative_personas'],
        'n_suffixes': len(suffixes), 'n_pairs': len(dataset), 'seconds': round(secs, 1),
        'layer_keys': 'repeng: key k = output of decoder block k = hidden_states[k+1]',
        'versions': {p: version(p) for p in ('repeng', 'transformers', 'torch', 'scikit-learn', 'numpy')},
    }
    json.dump(meta, open(f'{stem}.json', 'w'), indent=2)
    print(f'[{trait}] wrote {stem}.pt/.npz/.json ({secs:.0f}s)')
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--trait', default='agreeableness', help=f'one of {TRAITS} or "all"')
    ap.add_argument('--model-key', default=None, help='entry of models.yaml (default: the config\'s)')
    ap.add_argument('--model-path', default=None,
                    help='load weights/tokenizer from this local dir instead of the Hub id in models.yaml')
    ap.add_argument('--out', default='steering/vectors')
    a = ap.parse_args()
    traits = TRAITS if a.trait == 'all' else a.trait.split(',')

    _, model_key, mc = load_configs(traits[0], a.model_key)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train.py lines 30-42, unchanged: fp16, device_map="auto", pad id 0, ControlModel wrapper
    src = a.model_path or mc['model_name']
    if a.model_path:
        mc = {**mc, 'model_path': a.model_path}
    print(f'loading {src} ({mc["dtype"]})')
    # `dtype=` is transformers>=4.56's name for train.py's `torch_dtype=`
    model = AutoModelForCausalLM.from_pretrained(src, dtype=DTYPES[mc['dtype']], device_map='auto')
    tok = AutoTokenizer.from_pretrained(src)
    tok.pad_token_id = 0
    model = ControlModel(model, mc['control_layers'])

    for trait in traits:
        compute(trait, model, tok, model_key, mc, out_dir)


if __name__ == '__main__':
    main()

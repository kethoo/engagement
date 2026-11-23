"""Do the Steer2Adapt vectors move the baseline's prediction, and is the move article-specific?

No training of vectors yet. The fixed baseline line (linear head, layer 16, mean pooling,
weight decay 10 — the holdout's choice) is fit on the 85 non-test train articles. Then for
every article, trait vector ``d`` and strength ``α``, ``α·d`` is added at repeng layer ``key``
(< 15, the head's input is ``hidden_states[16]``) and the head's prediction is recomputed.

Reported per (key, trait, α), over articles, for empathy and distress:
  shift      mean of pred(α) − pred(0)
  spread     sd of that change across articles — 0 would mean the vector only adds a
             constant, i.e. it could be replaced by a per-person offset
  rank_r     Pearson of pred(α) with pred(0) — whether the article ranking survives

Sanity check first: with α = 0 the split forward must reproduce the cached baseline
features (cosine ≈ 1).

    python -m steering.dose_response --model-path ~/wassa_Luka/models/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from engagement.__main__ import load_config
from engagement.data import fit_additive
from engagement.features import FeatureStore, build_prompt, cache_path
from engagement.heads import TorchHead

from . import TRAITS
from .steered import SteeredLlama


def fit_head(cfg, essays, store, layer=16, wd=10.0):
    train = essays[essays.split == 'train']
    train = train[~train.article_key.isin(set(essays[essays.split == 'test'].article_key))]
    mu, a, _ = fit_additive(train, cfg['targets']['person_ridge'])
    T = (a + mu).sort_index()
    h = cfg['head']
    head = TorchHead(wd, 0, h['dropout'], h['lr'], h['epochs'], cfg['seed'])
    head.fit(store.X(list(T.index), layer, 'mean'), T[['empathy', 'distress']].to_numpy())
    return head.net.eval(), list(T.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/baseline.yaml')
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--vectors', default='steering/vectors')
    ap.add_argument('--keys', default='8,10,12,14')
    ap.add_argument('--alphas', default='-8,-4,-2,2,4,8')
    ap.add_argument('--method', default='pca_center', choices=['pca_center', 'diffmean'])
    ap.add_argument('--out', default='steering/results/dose_response.json')
    a = ap.parse_args()

    cfg = load_config(a.config)
    proc = Path(cfg['data']['processed_dir'])
    essays, articles = pd.read_csv(proc / 'essays.csv'), pd.read_csv(proc / 'articles.csv')
    store = FeatureStore(cache_path(cfg))
    head, head_keys = fit_head(cfg, essays, store)
    head = head.to('cuda')

    llm = SteeredLlama(a.model_path)
    arts = articles[articles.article_key.isin(list(store.keys))].reset_index(drop=True)
    ids = [llm.encode(build_prompt(llm.tok, r.title, r.text, cfg)[0]) for r in arts.itertuples()]
    keys, alphas = [int(k) for k in a.keys.split(',')], [float(x) for x in a.alphas.split(',')]

    vecs = {}
    for t in TRAITS:
        z = np.load(Path(a.vectors) / f'{t}_llama_3_1_8b_ins.npz')
        L = list(z['layers'])
        d = z[a.method]
        vecs[t] = {k: torch.tensor(d[L.index(k)] / np.linalg.norm(d[L.index(k)]), dtype=torch.float32) for k in keys}

    combos = [(t, al) for t in TRAITS for al in alphas]
    report = {'head': 'linear L16 mean wd=10, fit on 85 non-test train articles (adjusted target)',
              'method': a.method, 'alphas': alphas, 'n_articles': len(arts), 'keys': {}}
    t0 = time.time()
    for key in keys:
        V = torch.stack([vecs[t][key] * al for t, al in combos]).to('cuda')
        base, steered, norms, sanity = [], [], [], []
        for i, x in enumerate(ids):
            with torch.no_grad():
                h = llm.prefix(x, key)
                norms.append(h[0, 1:].float().norm(dim=-1).median().item())  # skip BOS (huge norm)
                h0 = llm.suffix(h, key, 16)[0].float().mean(0)
                cached = torch.tensor(store.X([arts.article_key[i]], 16, 'mean')[0], device='cuda')
                sanity.append(torch.nn.functional.cosine_similarity(h0, cached, dim=0).item())
                base.append(head(h0[None]).cpu().numpy()[0])
                out = []
                for c in range(0, len(V), 8):
                    v = V[c:c + 8]
                    hs = llm.suffix(h.expand(len(v), -1, -1), key, 16, v).float().mean(1)
                    out.append(head(hs).cpu().numpy())
                steered.append(np.concatenate(out))
        base, steered = np.array(base), np.stack(steered, 1)  # [N,2], [C,N,2]
        rows = {}
        for ci, (t, al) in enumerate(combos):
            delta = steered[ci] - base
            rows.setdefault(t, {})[str(al)] = {
                lab: {'shift': float(delta[:, j].mean()), 'spread': float(delta[:, j].std()),
                      'rank_r': float(np.corrcoef(steered[ci][:, j], base[:, j])[0, 1])}
                for j, lab in enumerate(['empathy', 'distress'])}
        report['keys'][key] = {'residual_norm_median': float(np.median(norms)),
                               'sanity_cos_min': float(np.min(sanity)), 'traits': rows}
        print(f'\n== key {key} (output of block {key}; residual norm ≈ {np.median(norms):.1f}; '
              f'sanity cos min {np.min(sanity):.5f}; {time.time() - t0:.0f}s)')
        print(f'  {"trait":17s} {"α":>5s} | {"emp shift":>9s} {"spread":>6s} {"rank r":>6s} | {"dis shift":>9s} {"spread":>6s} {"rank r":>6s}')
        for t in TRAITS:
            for al in alphas:
                e, d_ = rows[t][str(al)]['empathy'], rows[t][str(al)]['distress']
                print(f'  {t:17s} {al:+5.0f} | {e["shift"]:+9.3f} {e["spread"]:6.3f} {e["rank_r"]:6.3f} '
                      f'| {d_["shift"]:+9.3f} {d_["spread"]:6.3f} {d_["rank_r"]:6.3f}')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()

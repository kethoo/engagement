"""python -m engagement {prepare,features,train,all} --config configs/<name>.yaml

prepare   raw CSVs -> data/processed/{essays,articles}.csv + data_report.json (no GPU)
features  frozen-LLaMA hidden states per article -> cache/features/*.pt   (GPU)
train     nested article-grouped CV of the head on cached features -> runs/<name>_<time>/
all       the three in order
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from . import data


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ('lr',):  # YAML 1.1 reads 1e-3 as a string
        if k in cfg.get('head', {}):
            cfg['head'][k] = float(cfg['head'][k])
    cfg['head']['weight_decay'] = [float(x) for x in cfg['head'].get('weight_decay', [])]
    cfg['head']['alphas'] = [float(x) for x in cfg['head'].get('alphas', [])]
    return cfg


def prepare(cfg):
    out = Path(cfg['data']['processed_dir'])
    out.mkdir(parents=True, exist_ok=True)
    essays, articles = data.build(cfg['data']['raw_dir'])
    essays.to_csv(out / 'essays.csv', index=False)
    articles.to_csv(out / 'articles.csv', index=False)
    rep = data.report(essays, articles)
    lam = cfg['targets']['person_ridge']
    train = essays[essays.split == 'train']
    rep['target_reliability_train'] = {k: data.reliability(train, k, lam) for k in ('raw_mean', 'adjusted')}
    (out / 'data_report.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def load_processed(cfg):
    p = Path(cfg['data']['processed_dir'])
    return pd.read_csv(p / 'essays.csv'), pd.read_csv(p / 'articles.csv')


def features(cfg):
    from .features import extract

    _, articles = load_processed(cfg)
    extract(cfg, articles)


def train(cfg):
    from . import evaluate

    essays, articles = load_processed(cfg)
    store = None
    if cfg['head']['type'] != 'tfidf_ridge':
        from .features import FeatureStore, cache_path

        store = FeatureStore(cache_path(cfg))
        for k in ('layers', 'torch_layers'):
            if k in cfg['head']:
                cfg['head'][k] = [l for l in cfg['head'][k] if -store.n_layers - 1 <= l <= store.n_layers]
    views = evaluate.Views(store, articles)

    run = Path(cfg['output_dir']) / f'{cfg["name"]}_{cfg["head"]["type"]}_{time.strftime("%Y%m%d-%H%M%S")}'
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    results = {}
    for proto in cfg['eval']['protocols']:
        print(f'== {proto}')
        res, preds, chosen = getattr(evaluate, proto)(cfg, essays, views)
        results[proto] = {'metrics': res, 'chosen': chosen}
        preds.to_csv(run / f'predictions_{proto}.csv', index=False)
    (run / 'results.json').write_text(json.dumps(results, indent=2))

    if 'holdout' in results:
        m = results['holdout']['metrics']
        print(f'\nholdout — chose {m["chosen"]} (val r={m["val_score"]:.3f}); test, 90% CI over test articles:')
        for k, v in m['test'].items():
            print(f'  {k:28s} {v["value"]:+.3f}  [{v["ci90"][0]:+.3f}, {v["ci90"][1]:+.3f}]')
    if 'article_cv' in results:
        m = results['article_cv']['metrics']
        print('\narticle_cv on train essays (mean ± sd over repeats)')
        for k in m:
            print(f'  {k:28s} {m[k]["mean"]:+.3f} ± {m[k]["sd"]:.3f}')
    print(f'\nwrote {run}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'features', 'train', 'all'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--head', help='override head.type (ridge|linear|mlp|tfidf_ridge)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override a config value, e.g. data.exclude_test_articles_from_train=false')
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.head:
        cfg['head']['type'] = a.head
    for kv in a.set:
        key, val = kv.split('=', 1)
        node = cfg
        *path, leaf = key.split('.')
        for p in path:
            node = node[p]
        node[leaf] = yaml.safe_load(val)
        cfg['name'] += f'__{leaf}={val}'
    for stage in (['prepare', 'features', 'train'] if a.stage == 'all' else [a.stage]):
        if stage == 'features' and cfg['head']['type'] == 'tfidf_ridge':
            continue
        globals()[stage](cfg)


if __name__ == '__main__':
    main()

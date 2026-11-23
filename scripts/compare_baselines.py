"""Score Luka's zero-shot baseline and our head baselines with one evaluation.

Both are scored on the same articles, against the same targets, with the same metrics
and the same CI method, so the numbers can be put side by side:

  test   official test set: 14 articles / 83 essays. Target = raw mean of the test
         readers (they are all unseen persons); essay level = each essay's own label.
  train  the train articles both cover. Ours = out-of-fold predictions of the
         article-grouped CV (metrics averaged over its repeats); Luka's zero-shot never
         trains, so all its train predictions are out-of-sample. Targets: raw mean and
         the adjusted mean μ + a (person offsets removed, fit on all train essays).

Metrics per label: Pearson (+ 95% bootstrap CI over articles; essays move with their
article), Spearman, MAE, RMSE, and the mean predicted vs true level. Luka's baseline
predicts empathy only, so distress is reported for our heads alone.

Articles are matched by normalised title (the two projects number articles differently).

    python scripts/compare_baselines.py \
        --ours runs/baseline_article_only_linear_* runs/baseline_article_only_ridge_* \
        --luka-test ~/wassa_Luka/runs/zeroshot_test_20261118_012548 \
        --luka-train ~/wassa_Luka/runs/zeroshot_train_20261118_012609
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

import numpy as np
import pandas as pd
from scipy import stats

from engagement.data import LABELS, article_targets, fit_additive


def _norm(s) -> str:
    return re.sub(r'\W+', ' ', str(s).lower()).strip()


def metrics(pred: pd.Series, essays: pd.DataFrame, label: str, target: pd.Series) -> dict:
    """Article- and essay-level metrics of article predictions ``pred`` (index article_key)."""
    keys = pred.index.intersection(target.index)
    p, t = pred.loc[keys].to_numpy(float), target.loc[keys].to_numpy(float)
    e = essays[essays.article_key.isin(keys)]
    pe, te = pred.loc[e.article_key].to_numpy(float), e[label].to_numpy(float)
    out = {}
    for lvl, (x, y) in {'article': (p, t), 'essay': (pe, te)}.items():
        out[lvl] = {
            'n': len(x),
            'pearson': float(stats.pearsonr(x, y)[0]),
            'spearman': float(stats.spearmanr(x, y)[0]),
            'mae': float(np.abs(x - y).mean()),
            'rmse': float(np.sqrt(((x - y) ** 2).mean())),
            'mean_pred': float(x.mean()),
            'mean_true': float(y.mean()),
        }
    return out


def bootstrap_ci(pred, essays, label, target, n=2000, seed=0, level=0.95) -> dict:
    """CI of article/essay Pearson by resampling articles with replacement."""
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(pred.index.intersection(target.index)))
    groups = {k: essays.loc[essays.article_key == k, label].to_numpy(float) for k in keys}
    p, t = pred.loc[keys].to_numpy(float), target.loc[keys].to_numpy(float)
    art, ess = [], []
    for _ in range(n):
        i = rng.integers(0, len(keys), len(keys))
        art.append(np.corrcoef(p[i], t[i])[0, 1])
        ye = np.concatenate([groups[keys[j]] for j in i])
        xe = np.concatenate([np.full(len(groups[keys[j]]), p[j]) for j in i])
        ess.append(np.corrcoef(xe, ye)[0, 1])
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    return {'article': np.nanquantile(art, q).tolist(), 'essay': np.nanquantile(ess, q).tolist()}


def luka_preds(run_dir: Path, title_to_key: dict) -> pd.Series:
    df = pd.read_csv(run_dir / 'predictions_articles.csv')
    df['article_key'] = df.title.map(lambda s: title_to_key.get(_norm(s)))
    if df.article_key.isna().any():
        raise ValueError(f'unmatched titles in {run_dir}: {df.title[df.article_key.isna()].tolist()[:3]}')
    return df.groupby('article_key').pred.mean()  # the duplicated article appears twice


def head_name(run: Path) -> str:
    return re.search(r'_(tfidf_ridge|ridge|linear|mlp)_\d{8}-\d{6}$', run.name).group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ours', nargs='+', required=True, help='our run dirs (main variant)')
    ap.add_argument('--luka-test', required=True)
    ap.add_argument('--luka-train', required=True)
    ap.add_argument('--processed', default='data/processed')
    ap.add_argument('--person-ridge', type=float, default=1.0)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--out', default='runs/comparison_luka')
    a = ap.parse_args()

    essays = pd.read_csv(Path(a.processed) / 'essays.csv')
    articles = pd.read_csv(Path(a.processed) / 'articles.csv')
    title_to_key = {_norm(t): k for t, k in zip(articles.title, articles.article_key)}
    test, train = essays[essays.split == 'test'], essays[essays.split == 'train']

    T_test = article_targets(test, 'raw_mean')
    mu, a_, _ = fit_additive(train, a.person_ridge)
    T_train = {'raw': article_targets(train, 'raw_mean'), 'adjusted': (a_ + mu)}

    models_test = {'luka_zeroshot': {'empathy': luka_preds(Path(a.luka_test).expanduser(), title_to_key)}}
    luka_train = luka_preds(Path(a.luka_train).expanduser(), title_to_key)
    cv = {}
    for run in map(Path, a.ours):
        h = pd.read_csv(run / 'predictions_holdout.csv').set_index('article_key')
        models_test[head_name(run)] = {l: h[l] for l in LABELS}
        cv[head_name(run)] = pd.read_csv(run / 'predictions_article_cv.csv')

    report = {'test': {}, 'train': {}}
    for name, preds in models_test.items():
        for label, p in preds.items():
            m = metrics(p, test, label, T_test[label])
            ci = bootstrap_ci(p, test, label, T_test[label], a.bootstrap)
            for lvl in ('article', 'essay'):
                m[lvl]['pearson_ci95'] = ci[lvl]
            report['test'].setdefault(name, {})[label] = m

    # train: only the articles both cover; ours averaged over CV repeats
    common = sorted(set(luka_train.index) & set(next(iter(cv.values())).article_key))
    tr = train[train.article_key.isin(common)]
    for tname, T in T_train.items():
        T = T.loc[common]
        report['train'].setdefault('luka_zeroshot', {}).setdefault('empathy', {})[tname] = \
            metrics(luka_train.loc[common], tr, 'empathy', T['empathy'])
        for name, df in cv.items():
            for label in LABELS:
                reps = [metrics(g.set_index('article_key')[label].loc[common], tr, label, T[label])
                        for _, g in df.groupby('repeat')]
                avg = {lvl: {k: float(np.mean([r[lvl][k] for r in reps])) for k in reps[0][lvl]} for lvl in reps[0]}
                report['train'].setdefault(name, {}).setdefault(label, {})[tname] = avg
    report['train_articles'] = len(common)
    report['train_essays'] = len(tr)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out / 'comparison.json', 'w'), indent=2)

    # ---- one table per split, empathy first (the label both baselines predict)
    lines = [f'# Luka (zero-shot) vs head baselines — same articles, same metrics\n',
             '## Test: 14 articles / 83 essays, target = raw mean of the test readers\n',
             '| model | label | article r [95% CI] | essay r [95% CI] | article MAE | essay MAE | mean pred / true |',
             '|---|---|---|---|---|---|---|']
    for label in LABELS:
        for name, r in report['test'].items():
            if label not in r:
                continue
            x, e = r[label]['article'], r[label]['essay']
            lines.append(f'| {name} | {label} | {x["pearson"]:.3f} [{x["pearson_ci95"][0]:.2f}, {x["pearson_ci95"][1]:.2f}] '
                         f'| {e["pearson"]:.3f} [{e["pearson_ci95"][0]:.2f}, {e["pearson_ci95"][1]:.2f}] '
                         f'| {x["mae"]:.2f} | {e["mae"]:.2f} | {x["mean_pred"]:.2f} / {x["mean_true"]:.2f} |')
    lines += ['', f'## Train: {len(common)} articles / {len(tr)} essays (ours: out-of-fold CV, mean over repeats)\n',
              '| model | label | article r (raw mean) | article r (adjusted) | essay r | article MAE (raw) | essay MAE |',
              '|---|---|---|---|---|---|---|']
    for label in LABELS:
        for name, r in report['train'].items():
            if label not in r:
                continue
            raw, adj = r[label]['raw'], r[label]['adjusted']
            lines.append(f'| {name} | {label} | {raw["article"]["pearson"]:.3f} | {adj["article"]["pearson"]:.3f} '
                         f'| {raw["essay"]["pearson"]:.3f} | {raw["article"]["mae"]:.2f} | {raw["essay"]["mae"]:.2f} |')
    md = '\n'.join(lines) + '\n'
    open(out / 'comparison.md', 'w').write(md)
    print(md)
    print(f'wrote {out}/comparison.json, comparison.md')


if __name__ == '__main__':
    main()

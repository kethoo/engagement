"""Evaluation protocols for the article-only ("average reader") regressor.

The official dev split is not used anywhere.

``holdout`` (the planned protocol)
    train  = official train essays, minus (by default) the 14 test articles — every
             test article is also a train article, and keeping them would test recall
    val    = ``data.val_articles`` whole train articles (~10 essays each, so 10 ≈ 100
             essays), drawn with ``data.val_seed``. Whole articles, because the model sees
             only the article: random essays would put val articles in train too.
    select = the candidate (layer, pooling, regularisation) with the best val score,
             then refit on train+val
    test   = official test essays; test persons are all unseen, so the test target is
             the raw mean. 90% CIs by bootstrapping test articles (only 14).

``article_cv`` (stability check, train essays only)
    Nested, article-grouped K-fold over the train essays. Hyper-parameters are picked by
    an inner article-grouped CV on the training fold only; person effects for targets
    are fit on training-fold essays only. Repeated; mean ± sd over repeats. With a
    10-article val set the holdout's selection and score are noisy — this is the
    lower-variance estimate of the same quantity.

Metrics: article-level Pearson against the adjusted target and against the raw mean,
and essay-level Pearson (article prediction vs each essay's own label). ``avg`` is the
mean over empathy and distress, the official Track 3 aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABELS, article_targets, fit_additive
from .heads import candidates


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _folds(keys, k, rng):
    keys = np.array(sorted(keys))
    perm = rng.permutation(len(keys))
    return [keys[perm[i::k]] for i in range(k)]


class Views:
    """Turns a candidate's view into a design matrix for a list of article keys."""

    def __init__(self, store, articles: pd.DataFrame):
        self.store = store
        self.text = dict(zip(articles.article_key, articles.title.fillna('') + '\n' + articles.text))

    def X(self, keys, view):
        if view == 'text':
            return [self.text[k] for k in keys]
        return self.store.X(keys, *view)


def _select(cands, views, T: pd.DataFrame, k: int, rng) -> tuple:
    """Inner CV on the training articles' targets ``T``; returns the best candidate."""
    keys = list(T.index)
    folds = _folds(keys, k, rng)
    scores = []
    for name, view, make in cands:
        pred = pd.DataFrame(index=T.index, columns=LABELS, dtype=float)
        for f in folds:
            held = set(f)
            tr = [x for x in keys if x not in held]
            m = make().fit(views.X(tr, view), T.loc[tr, LABELS].to_numpy())
            pred.loc[f] = m.predict(views.X(list(f), view))
        scores.append(np.nanmean([pearson(pred[l], T[l]) for l in LABELS]))
    best = int(np.nanargmax(scores))
    return cands[best], scores[best]


def _metrics(pred: pd.DataFrame, essays: pd.DataFrame, targets: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for tname, T in targets.items():
        common = pred.index.intersection(T.index)
        for l in LABELS:
            out[f'article_{tname}_{l}'] = pearson(pred.loc[common, l], T.loc[common, l])
        out[f'article_{tname}_avg'] = float(np.mean([out[f'article_{tname}_{l}'] for l in LABELS]))
    e = essays.join(pred, on='article_key', rsuffix='_pred')
    for l in LABELS:
        out[f'essay_{l}'] = pearson(e[f'{l}_pred'], e[l])
    out['essay_avg'] = float(np.mean([out[f'essay_{l}'] for l in LABELS]))
    return out


def article_cv(cfg, essays, views) -> tuple[dict, pd.DataFrame, list]:
    ev, kind, lam = cfg['eval'], cfg['targets']['kind'], cfg['targets']['person_ridge']
    essays = essays[essays.split == 'train']
    cands = candidates(cfg)
    rows, preds, chosen = [], [], []
    for r in range(ev['repeats']):
        rng = np.random.default_rng(cfg['seed'] + r)
        parts, T_adj_eval, T_raw_eval = [], [], []
        for fi, test_keys in enumerate(_folds(essays.article_key.unique(), ev['outer_folds'], rng)):
            te = essays[essays.article_key.isin(test_keys)]
            tr = essays[~essays.article_key.isin(test_keys)]
            mu, a, b = fit_additive(tr, lam)
            T_train = (a + mu) if kind == 'adjusted' else article_targets(tr, 'raw_mean')
            (name, view, make), inner = _select(cands, views, T_train, ev['inner_folds'], rng)
            m = make().fit(views.X(list(T_train.index), view), T_train[LABELS].to_numpy())
            keys = sorted(test_keys)
            parts.append(pd.DataFrame(m.predict(views.X(keys, view)), index=keys, columns=LABELS))
            T_adj_eval.append(article_targets(te, 'adjusted', mu=mu, b=b))
            T_raw_eval.append(article_targets(te, 'raw_mean'))
            chosen.append({'repeat': r, 'fold': fi, 'choice': name, 'inner_score': inner})
            print(f'  repeat {r} fold {fi}: chose {name} (inner avg r={inner:.3f})')
        oof = pd.concat(parts).rename_axis('article_key')
        res = _metrics(oof, essays, {'adjusted': pd.concat(T_adj_eval), 'raw': pd.concat(T_raw_eval)})
        rows.append(res)
        preds.append(oof.assign(repeat=r))
        print(f'repeat {r}: ' + '  '.join(f'{k}={v:.3f}' for k, v in res.items() if k.endswith('avg')))
    df = pd.DataFrame(rows)
    summary = {k: {'mean': float(df[k].mean()), 'sd': float(df[k].std(ddof=1)) if len(df) > 1 else 0.0} for k in df}
    return summary, pd.concat(preds).reset_index(), chosen


def _train_targets(essays, kind, lam):
    mu, a, b = fit_additive(essays, lam)
    return ((a + mu) if kind == 'adjusted' else article_targets(essays, 'raw_mean')), b


def _predict(model, views, keys, view):
    keys = sorted(keys)
    return pd.DataFrame(model.predict(views.X(keys, view)), index=pd.Index(keys, name='article_key'), columns=LABELS)


def _bootstrap_ci(pred, essays, targets, n, seed, level=0.90):
    """CI of each metric by resampling whole test articles (essays move with their article)."""
    rng = np.random.default_rng(seed)
    groups = essays.groupby('article_key').indices
    arts = np.array(sorted(groups))
    samples = []
    for _ in range(n):
        pick = rng.choice(arts, len(arts))
        # relabel duplicates so a resampled article counts as a distinct one
        e = pd.concat([essays.iloc[groups[k]].assign(article_key=f'{k}#{i}') for i, k in enumerate(pick)])
        p = pd.DataFrame(pred.loc[pick].to_numpy(), index=[f'{k}#{i}' for i, k in enumerate(pick)], columns=LABELS)
        t = pd.DataFrame(targets.loc[pick].to_numpy(), index=p.index, columns=LABELS)
        samples.append(_metrics(p, e, {'raw': t}))
    s = pd.DataFrame(samples)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {k: [float(s[k].quantile(lo)), float(s[k].quantile(hi))] for k in s}


def holdout(cfg, essays, views) -> tuple[dict, pd.DataFrame, list]:
    d, ev = cfg['data'], cfg['eval']
    kind, lam = cfg['targets']['kind'], cfg['targets']['person_ridge']
    train, test = essays[essays.split == 'train'], essays[essays.split == 'test']
    test_arts = set(test.article_key)
    # val is always drawn from the non-test train articles, so the exclude/keep variants
    # share the same val set and differ only in whether test articles are trained on
    arts = np.array(sorted(set(train.article_key) - test_arts))
    val_arts = set(np.random.default_rng(d['val_seed']).choice(arts, d['val_articles'], replace=False))
    if d.get('exclude_test_articles_from_train', True):
        train = train[~train.article_key.isin(test_arts)]
    val, tr = train[train.article_key.isin(val_arts)], train[~train.article_key.isin(val_arts)]
    print(f'holdout: train {tr.article_key.nunique()} articles / {len(tr)} essays, '
          f'val {len(val_arts)} / {len(val)}, test {test.article_key.nunique()} / {len(test)}')

    T_tr, b = _train_targets(tr, kind, lam)
    T_val = article_targets(val, kind if kind == 'raw_mean' else 'adjusted', b=b)
    cands, scores = candidates(cfg), []
    for name, view, make in cands:
        p = _predict(make().fit(views.X(list(T_tr.index), view), T_tr[LABELS].to_numpy()), views, T_val.index, view)
        scores.append({'choice': name, 'val_score': float(np.nanmean([pearson(p[l], T_val[l]) for l in LABELS]))})
    i = int(np.nanargmax([x['val_score'] for x in scores]))
    best, (_, best_view, best_make) = scores[i], cands[i]
    print(f'holdout: chose {best["choice"]} (val r={best["val_score"]:.3f}); '
          f'val r over all {len(scores)} candidates: median {np.nanmedian([x["val_score"] for x in scores]):.3f}')

    T_fit = _train_targets(train, kind, lam)[0] if ev.get('refit_on_train_val', True) else T_tr
    model = best_make().fit(views.X(list(T_fit.index), best_view), T_fit[LABELS].to_numpy())
    pred = _predict(model, views, test.article_key.unique(), best_view)
    T_test = article_targets(test, 'raw_mean')  # test persons are unseen: offset 0
    res = {k: v for k, v in _metrics(pred, test, {'raw': T_test}).items()}
    ci = _bootstrap_ci(pred, test, T_test, ev.get('bootstrap', 1000), cfg['seed'])
    out = {
        'chosen': best['choice'], 'val_score': best['val_score'],
        'test': {k: {'value': v, 'ci90': ci[k]} for k, v in res.items()},
        'n': {'train_articles': int(tr.article_key.nunique()), 'val_articles': len(val_arts),
              'test_articles': int(test.article_key.nunique()), 'test_essays': len(test)},
        'val_articles': sorted(val_arts),
    }
    return out, pred.reset_index(), scores

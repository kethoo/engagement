"""WASSA 2024 Track 3 (EMP) data: essays, the news articles they react to, and targets.

Three properties of the raw release that the rest of the code relies on being handled here:

1. ``article_id`` is a *per-table* id. ``ec_article_info_train.csv`` and
   ``ec_article_info_dev.csv`` number the same articles differently (id 1 is a different
   article in each). Train essays resolve against the train table; dev and test essays
   against the dev table (verified by essay/article TF-IDF matching). Every article is
   mapped to a canonical key by its normalised text, which shows every dev/test
   article is also a train article.
2. ``person_id`` is also per split: train ``p033`` and dev ``33`` have different
   demographics in the Track 4 files. Persons are keyed as ``"<split>:<id>"``.
3. Labels are dominated by the person, not the article (person explains ~70% of
   essay-label variance, article ~10%, about what chance gives with 10 essays per
   article). The "average reader" target therefore removes person effects with an
   additive model rather than taking a raw per-article mean.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ['empathy', 'distress']

# The official dev split (track3_dev.csv) is deliberately not used: validation is carved
# out of train (whole articles), test is the official test set.
ESSAY_FILES = {
    'train': ('track3_train.csv', 'articles/ec_article_info_train.csv'),
    'test': ('test/test_data_with_labels/goldstandard_EMP.csv', 'articles/ec_article_info_dev.csv'),
}
ARTICLE_TABLES = ['articles/ec_article_info_train.csv', 'articles/ec_article_info_dev.csv']


def _read(path: Path) -> pd.DataFrame:
    # Parsing recipe given on the competition page.
    return pd.read_csv(path, escapechar='\\')


def _norm_text(s: str) -> str:
    return re.sub(r'\W+', ' ', str(s).lower()).strip()


def build(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(essays, articles)``.

    essays:   one row per labelled essay — split, essay_idx, person, article_key, essay,
              empathy, distress.
    articles: one row per canonical article — article_key, title, text.
    """
    raw = Path(raw_dir)

    # Canonical article key = first train-table article_id with that normalised text —
    # or with that title: train ids 247/413 are two versions of one article (TF-IDF
    # cosine 0.99) and would otherwise leak across CV folds.
    canon_by_text: dict[str, str] = {}
    canon_by_title: dict[str, str] = {}
    article_rows = {}
    for table in ARTICLE_TABLES:
        for row in _read(raw / table).itertuples():
            t, ti = _norm_text(row.text), _norm_text(row.title)
            if t not in canon_by_text:
                key = canon_by_title.get(ti) or (
                    f'a{int(row.article_id):03d}' if 'train' in table else f'dev_a{int(row.article_id):03d}'
                )
                canon_by_text[t] = key
                canon_by_title.setdefault(ti, key)
                article_rows.setdefault(key, {'article_key': key, 'title': row.title, 'text': row.text})

    frames = []
    for split, (essay_file, article_table) in ESSAY_FILES.items():
        df = _read(raw / essay_file)
        table = _read(raw / article_table)
        id_to_key = {int(r.article_id): canon_by_text[_norm_text(r.text)] for r in table.itertuples()}
        missing = set(df.article_id) - set(id_to_key)
        if missing:
            raise ValueError(f'{split}: article ids without text: {sorted(missing)[:10]}')
        frames.append(pd.DataFrame({
            'split': split,
            'essay_idx': np.arange(len(df)),
            'person': [f'{split}:{str(p).lstrip("p").lstrip("0") or "0"}' for p in df.person_id],
            'article_key': df.article_id.map(id_to_key),
            'essay': df.person_essay,
            'empathy': df.person_empathy.astype(float),
            'distress': df.person_distress.astype(float),
        }))
    essays = pd.concat(frames, ignore_index=True)
    articles = pd.DataFrame([article_rows[k] for k in sorted(essays.article_key.unique())])
    return essays, articles


# --------------------------------------------------------------------------- targets


def fit_additive(essays: pd.DataFrame, lam: float = 1.0, iters: int = 100):
    """Fit ``y = mu + a[article] + b[person] + e`` per label by ridge-shrunk alternating LS.

    Returns ``mu`` (Series over LABELS), ``a`` (DataFrame article_key x LABELS) and
    ``b`` (DataFrame person x LABELS). ``mu + a`` is the score an average person would
    give the article; ``b`` is the person's offset — what persona steering should add.
    """
    y = essays[LABELS]
    mu = y.mean()
    b = pd.DataFrame(0.0, index=pd.Index(essays.person.unique(), name='person'), columns=LABELS)
    for _ in range(iters):
        r = y - mu - b.loc[essays.person].to_numpy()
        g = r.groupby(essays.article_key.to_numpy())
        a = g.sum() / (g.count() + lam)
        r = y - mu - a.loc[essays.article_key].to_numpy()
        g = r.groupby(essays.person.to_numpy())
        b_new = g.sum() / (g.count() + lam)
        delta = (b_new - b.loc[b_new.index]).abs().to_numpy().max()
        b = b_new
        if delta < 1e-8:
            break
    a.index.name, b.index.name = 'article_key', 'person'
    return mu, a, b


def article_targets(essays: pd.DataFrame, kind: str, mu=None, b=None) -> pd.DataFrame:
    """Per-article target (article_key x LABELS).

    ``raw_mean``: mean essay label. ``adjusted``: mean of ``y - b[person]`` with person
    offsets from a model fit elsewhere (persons it has not seen get offset 0), i.e. an
    estimate of ``mu + a[article]`` that does not use the essays' own person effects.
    """
    y = essays[LABELS]
    if kind == 'adjusted':
        if b is None:
            mu, a, b = fit_additive(essays)
            return (a + mu).sort_index()
        offs = b.reindex(essays.person).fillna(0.0).to_numpy()
        y = y - offs
    elif kind != 'raw_mean':
        raise ValueError(f'unknown target kind {kind!r}')
    return y.groupby(essays.article_key.to_numpy()).mean().rename_axis('article_key').sort_index()


def reliability(essays: pd.DataFrame, kind: str, lam: float, n_splits: int = 200, seed: int = 0) -> dict:
    """Split-half reliability of article targets, Spearman-Brown corrected.

    Essays of each article are split in half; targets from each half are correlated
    across articles. ``sqrt(reliability)`` is the ceiling on the article-level Pearson
    any model can reach against these targets. For ``adjusted``, each half's person
    offsets come from the *other* half (cross-fitted), as in the CV evaluation —
    in-sample offsets inflate reliability (0.65 vs 0.58 for empathy).
    """
    rng = np.random.default_rng(seed)
    out = {l: [] for l in LABELS}
    for _ in range(n_splits):
        half = np.zeros(len(essays), bool)
        for _, idx in essays.groupby('article_key').indices.items():
            idx = rng.permutation(idx)
            half[idx[: len(idx) // 2]] = True
        A, B = essays[half], essays[~half]
        if kind == 'adjusted':
            t1 = article_targets(A, kind, b=fit_additive(B, lam)[2])
            t2 = article_targets(B, kind, b=fit_additive(A, lam)[2])
        else:
            t1, t2 = article_targets(A, kind), article_targets(B, kind)
        common = t1.index.intersection(t2.index)
        for l in LABELS:
            out[l].append(np.corrcoef(t1.loc[common, l], t2.loc[common, l])[0, 1])
    res = {}
    for l in LABELS:
        r = float(np.mean(out[l]))
        sb = 2 * r / (1 + r) if r > -1 else float('nan')
        res[l] = {'split_half_r': r, 'reliability': sb, 'pearson_ceiling': float(np.sqrt(max(sb, 0)))}
    return res


def report(essays: pd.DataFrame, articles: pd.DataFrame) -> dict:
    """Facts about the release that decide what can be evaluated honestly."""
    by_split = {s: set(g.article_key) for s, g in essays.groupby('split')}
    persons = {s: set(g.person) for s, g in essays.groupby('split')}
    return {
        'essays_per_split': essays.split.value_counts().to_dict(),
        'unique_articles': int(essays.article_key.nunique()),
        'articles_per_split': {s: len(v) for s, v in by_split.items()},
        'test_articles_in_train': len(by_split['test'] & by_split['train']),
        'persons_per_split': {s: len(v) for s, v in persons.items()},
        'train_persons_with_>=10_essays': int((essays[essays.split == 'train'].person.value_counts() >= 10).sum()),
        'article_words': essays.drop_duplicates('article_key')
        .merge(articles, on='article_key')
        .text.str.split().str.len()
        .describe()[['min', '50%', 'max']]
        .to_dict(),
    }

# engagement

Predict how much **empathy** and **distress** a news article evokes in its readers, from
the article alone, with a frozen decoder LLM (LLaMA) and a regression head — then
personalise that prediction with steering vectors.

## Plan

1. **Baseline (this code)** — no personalisation. Article text → frozen LLaMA →
   regression head → the score an *average* reader gives.
2. **Steering** — add steering vectors to the frozen model's residual stream and compare:
   - one vector **per persona** (person), vs
   - one **shared** vector for all personas.

## Dataset

This project uses the data from the
**[WASSA 2024 Shared Task](https://codalab.lisn.upsaclay.fr/competitions/18810)**
(CodaLab competition 18810): Track 3 (EMP, essay-level empathy/distress) and Track 4
(PER, the writers' personality), plus the news-article texts.

The dataset is not redistributed here — it is obtained from the competition page,
under that competition's own terms and conditions. `scripts/download_data.sh` fetches it
into `data/raw/` (gitignored).

### Properties of the release that shape the experiment

Checked on the raw files (`python -m engagement prepare` writes `data/processed/data_report.json`):

- **`article_id` is per table.** The train and dev article files number articles
  differently. Train essays resolve against `ec_article_info_train.csv`, dev *and test*
  essays against `ec_article_info_dev.csv` (confirmed by essay↔article TF-IDF matching).
  Articles are keyed by normalised text.
- **`person_id` is per split.** Train `p033` and dev `33` have different demographics.
  Persons are keyed `"<split>:<id>"`; no person appears in more than one split.
- **One article appears twice** under different ids (train 247/413, dev 66/99 — "Paris
  attacks: A year of grief…", TF-IDF cosine 0.99). Merged by title, else it leaks across
  CV folds.
- **Every dev and test article is also a train article** (99 unique articles overall).
  The official train → dev/test split therefore never shows the model an unseen article.
- **The person dominates the label.** Person explains ~70% of essay-label variance; the
  article ~10%, about what chance gives with 10 essays per article. A raw per-article
  mean is a poor "average reader" target (split-half reliability 0.13 empathy / 0.26
  distress, and very imprecise: 90% bootstrap CI of the ceiling 0.11–0.57 for empathy).
  Removing person offsets with an additive model
  `y = μ + a[article] + b[person] + ε` raises it to 0.56 / 0.64 (cross-fitted).
  `μ + a` is the baseline's target; `b` is what persona steering has to supply.

## Pipeline

Plain PyTorch + 🤗 Transformers. Because the backbone is frozen, each article's hidden
states (all layers, mean- and last-token pooled) are computed once and cached; heads
train on the cache in seconds.

```bash
pip install -r requirements.txt
export HF_TOKEN=...                      # Llama-3.1 is gated on the Hub
bash scripts/run_baseline.sh             # download → prepare → features → all heads
```

or stage by stage:

```bash
python -m engagement prepare  --config configs/baseline.yaml   # CPU
python -m engagement features --config configs/baseline.yaml   # GPU, ~100 forward passes
python -m engagement train    --config configs/baseline.yaml --head ridge   # ridge|linear|mlp|tfidf_ridge
python -m engagement all      --config configs/smoke.yaml      # CPU smoke test, tiny random LLaMA
```

| Module | Role |
|---|---|
| `engagement/data.py` | load + fix ids, additive person/article model, targets, reliability |
| `engagement/features.py` | frozen-LLaMA hidden states per article → `cache/features/*.pt` |
| `engagement/heads.py` | `RidgeHead`, `TorchHead` (linear / MLP `nn.Module`), `TfidfRidgeHead` |
| `engagement/evaluate.py` | `article_cv` and `official` protocols, Pearson metrics |

### Evaluation

The official **dev split is not used**.

- **`holdout`** (the planned protocol): train = official train essays minus the 14 test
  articles (`data.exclude_test_articles_from_train`); val = 10 whole train articles
  (≈100 essays — whole articles, since the model only sees the article); the candidate
  (layer, pooling, regularisation) with the best val score is refit on train+val and
  scored on the official test set (83 essays, 14 articles). Test persons are all unseen,
  so the test target is the raw mean. 90% CIs by resampling test articles.
- **`article_cv`** (stability check): nested, article-grouped 5-fold CV over the train
  essays only, repeated 5×; selection by inner CV, person offsets fit on training folds.
  With 10 val and 14 test articles the holdout is very noisy (TF-IDF scores 0.39 on test
  but ≈0 in CV) — CV is the lower-variance estimate of the same quantity.

Metrics: Pearson at article level (vs adjusted / raw targets) and essay level (article
prediction vs each essay's label, as the WASSA scorer does); `avg` = mean of empathy and
distress, the official Track 3 aggregation. `--set key=value` overrides any config value.

Each run writes `runs/<name>_<head>_<time>/` with `config.yaml`, `results.json`
(metrics + the hyper-parameters chosen per fold) and out-of-fold predictions.

### Baseline results (2026-11-18, Llama-3.1-8B-Instruct, frozen)

Pearson, `avg` of empathy and distress. CV = article-grouped 5×5 CV on train (mean ± sd
over repeats); test = official test set, 90% CI over its 14 articles.

| head | CV, article (adjusted) | test, essay [90% CI] | test, article | chosen on val |
|---|---|---|---|---|
| ridge | **0.484 ± 0.078** | 0.261 [0.04, 0.38] | 0.413 | layer 16, mean |
| MLP | **0.480 ± 0.071** | 0.317 [0.11, 0.41] | 0.503 | layer 16, mean |
| linear | 0.425 ± 0.064 | 0.248 [0.03, 0.38] | 0.404 | layer 16, mean |
| TF-IDF (no LLM) | −0.024 ± 0.127 | 0.288 [0.07, 0.40] | 0.392 | — |

- Frozen LLaMA features carry the article signal (CV ≈ 0.48 against a ceiling of
  ≈ 0.75–0.80); bag-of-words does not (≈ 0).
- The 14-article test set cannot separate the models — TF-IDF is ≈ 0 in CV yet 0.29 on
  test. Report CV as the headline, test with its CI.
- All heads independently pick layer 16 / mean pooling — the natural layer for steering.
- **Training on the test articles makes test worse** (essay 0.06–0.08 for the LLaMA heads,
  same val set): for those articles the train readers' average barely predicts the test
  readers' (r = 0.04 empathy, 0.45 distress), so memorising it hurts. Hence the default
  `exclude_test_articles_from_train: true`.

Experiments are meant to move onto
**[micm-nlp](https://nlp.micm.edu.ge/en/latest/index.html)** (Muskhelishvili Institute of
Computational Mathematics, Georgian Technical University) once the setup is settled;
`RegressionHead` is a plain `nn.Module` so it can be wrapped as a `@micm_plugin` model.

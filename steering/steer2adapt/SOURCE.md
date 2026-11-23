Copied unchanged from [JackyJiang08/NeurIPS-2026-Steer2Adapt](https://github.com/JackyJiang08/NeurIPS-2026-Steer2Adapt)
at commit `c1bcb54` (MIT, see `LICENSE`):

- `configs/{openness,conscientiousness,extraversion,agreeableness,neuroticism}.yaml` — the persona
  prompts (template `"Act as if you're {persona}."`, 3 positive + 3 negative personas per trait,
  `method: pca_center`, `batch_size: 32`)
- `configs/models.yaml` — chat tags and control layers per model
- `data/all_truncated_outputs.json` — the 582 completion suffixes

Their pre-computed vectors (`vectors/*_llama_3_1_8b_ins.pt`) are not copied; `compare_vectors.py`
reads them from a clone of that repository.

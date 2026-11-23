"""Check our repeng vectors against the ones Steer2Adapt published.

Per layer: cosine of our ``pca_center`` direction with theirs (≈ 1 if the reproduction
worked; the sign is fixed by repeng, so −1 would also be a real disagreement), and of our
difference-of-means direction with theirs (how close the cheaper estimator is).

The published ``.pt`` files are pickles of ``repeng.extract.ControlVector``; they are read
with a restricted unpickler (numpy arrays only, no code from the file is executed) and
without needing repeng installed.

    python -m steering.compare_vectors --theirs ~/NeurIPS-2026-Steer2Adapt/vectors --trait agreeableness
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

from . import TRAITS

_NUMPY_GLOBALS = {
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'), ('_codecs', 'encode'),
}


class _ControlVector:
    """Stand-in for ``repeng.extract.ControlVector`` (attributes model_type, directions)."""


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('repeng.extract', 'ControlVector'):
            return _ControlVector
        if (module, name) in _NUMPY_GLOBALS:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f'refusing to load {module}.{name}')


def load_pt(path: str | Path) -> dict[int, np.ndarray]:
    """``torch.save(ControlVector)`` file -> {layer: direction}."""
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.endswith('/data.pkl'))
        obj = _SafeUnpickler(z.open(member)).load()
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in obj.directions.items()}


def cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare(trait: str, ours_dir: Path, theirs_dir: Path, model_key: str) -> dict:
    ours = np.load(ours_dir / f'{trait}_{model_key}.npz')
    layers = [int(l) for l in ours['layers']]
    refs = sorted(theirs_dir.glob(f'{trait}_{model_key}*.pt'))
    if not refs:
        raise FileNotFoundError(f'no {trait}_{model_key}*.pt in {theirs_dir}')
    out = {}
    for ref in refs:
        th = load_pt(ref)
        rows = {l: {'pca_center': cos(ours['pca_center'][i], th[l]),
                    'diffmean': cos(ours['diffmean'][i], th[l])}
                for i, l in enumerate(layers) if l in th}
        pc = np.array([r['pca_center'] for r in rows.values()])
        out[ref.name] = {
            'layers': rows,
            'pca_center_cos': {'min': float(pc.min()), 'median': float(np.median(pc)), 'max': float(pc.max())},
            'layers_missing_in_theirs': sorted(set(layers) - set(th)),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--theirs', required=True, help='Steer2Adapt vectors/ directory')
    ap.add_argument('--ours', default='steering/vectors')
    ap.add_argument('--trait', default='agreeableness', help=f'one of {TRAITS}, comma list, or "all"')
    ap.add_argument('--model-key', default='llama_3_1_8b_ins')
    ap.add_argument('--show-layers', default='1,5,10,15,20,22,25,31',
                    help='repeng layer keys to print (15 = baseline head layer 16, 22 = Steer2Adapt control layer)')
    a = ap.parse_args()
    traits = TRAITS if a.trait == 'all' else a.trait.split(',')
    show = [int(x) for x in a.show_layers.split(',')]

    report = {}
    for trait in traits:
        if not (Path(a.ours) / f'{trait}_{a.model_key}.npz').exists():
            print(f'{trait}: not computed yet, skipped')
            continue
        report[trait] = compare(trait, Path(a.ours), Path(a.theirs).expanduser(), a.model_key)
        for ref, r in report[trait].items():
            s = r['pca_center_cos']
            print(f'\n{trait} vs {ref}: cos(ours pca_center, theirs) over {len(r["layers"])} layers '
                  f'min {s["min"]:+.4f}  median {s["median"]:+.4f}  max {s["max"]:+.4f}')
            print(f'  {"layer":>5s}  {"pca_center":>10s}  {"diffmean":>9s}')
            for l in show:
                if l in r['layers']:
                    x = r['layers'][l]
                    print(f'  {l:5d}  {x["pca_center"]:+10.4f}  {x["diffmean"]:+9.4f}')
    out = Path(a.ours) / 'comparison.json'
    json.dump(report, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

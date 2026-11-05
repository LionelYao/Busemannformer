# Busemannformer

Horospherical self-attention for hyperbolic Transformers: directed and
bidirectional Busemann scores, log-depth modulation, sparse attention, and
learned boundary-dictionary attention.

Repository: [LionelYao/Busemannformer](https://github.com/LionelYao/Busemannformer).

See [attribution/licensing](NOTICE.md) for attribution and license status.

## Install and test

Python 3.10+; Python 3.12 is recommended for the recorded test environment.
Install an appropriate PyTorch build for your machine, then:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m examples.attention
```

No benchmark download, account, cluster, or GPU is required for the CPU tests.
Some historical GPU-specific tests are skipped on unsupported hardware.

## Train and evaluate

Inputs are numeric NPZ arrays with **already-preprocessed features**. There is
no automatic feature normalization, dataset conversion, or download.

- NC: `features[N,F]`, `edges[E,2]`, `labels[N]`, and disjoint, exhaustive
  `train_idx`, `val_idx`, `test_idx` arrays. This is transductive classification.
- LP: only `features[N,F]` and `edges[E,2]`. Node-label arrays are rejected.
  Validation/test edges are removed before constructing message-passing support.

Use zero-based integer node IDs. NPZ loading disables pickle.

```bash
python train_nc.py train --input data/nc.npz --config configs/nc_example.json --seed 42 --output outputs/nc42
python train_nc.py evaluate --input data/nc.npz --run outputs/nc42

python train_lp.py train --input data/lp.npz --config configs/lp_example.json --seed 42 --output outputs/lp42
python train_lp.py evaluate --input data/lp.npz --run outputs/lp42
```

Both runners select checkpoints on validation metrics; test scoring requires
the separate `evaluate` command. New output paths are required. Checkpoint,
input, source, and configuration hashes guard evaluation against accidental
changes. Use `--device cuda` for a CUDA run. Repeat with distinct seeds and
retain every run; do not select seeds by their test scores.

For a short, synthetic end-to-end check:

```bash
python -m examples.make_toy_data --output data/toy
python train_nc.py train --input data/toy/nc.npz --config configs/nc_smoke.json --output outputs/toy_nc
python train_nc.py evaluate --input data/toy/nc.npz --run outputs/toy_nc
python train_lp.py train --input data/toy/lp.npz --config configs/lp_smoke.json --output outputs/toy_lp
python train_lp.py evaluate --input data/toy/lp.npz --run outputs/toy_lp
```

## Attention mechanisms

`busemann_transformer.py` exposes `BusemannMHA` and `BusGraphTransformerNC`.
`paper_lp_model.py` provides directed/bidirectional score controls and
Fermi--Dirac decoders with distance power 1 or 2.

| Mechanism | Interpretation |
|---|---|
| Dense HSA | Original pairwise Busemann score; quadratic in token count |
| Sparse HSA | Exact on a specified support; cost depends on allowed pairs |
| Boundary dictionary | Exact associative evaluation of a different finite-rank kernel |
| Random features / landmarks | Experimental approximations; not the dictionary mechanism |

The dictionary is linear in token count only for fixed dictionary size and
width. It is not an exact finite-feature factorization of unrestricted HSA.

Code uses sectional curvature `-c`; the manuscript's `-c^2` convention therefore
corresponds to `c_code = c_paper^2`. Epsilon floors and raw-log scores are retained
as implemented, not silently replaced by the ideal mathematical expressions.

## Contents

- Model, split utilities, and focused regression tests with source provenance.
- Portable NC/LP runners, examples, and explicit configurations.
- SHA-256 release manifest and local archive/check utilities.
- `paper/checklist.tex`: manuscript include with the original author-selected
  answer labels and revised justifications; not a standalone LaTeX document.
- No datasets, checkpoints, cluster scripts, credentials, or experiment logs.

Run `python scripts/check_release.py` to verify the packaged files. The original
project license still needs to be located; no new license is assigned here.

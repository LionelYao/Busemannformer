# Release checks — 2026-11-04

- `python -m pytest -q`: **235 passed, 18 skipped** on macOS CPU.
  The skipped cases require CUDA or the historical H200 test environment.
- Dense LD-HSA and boundary-dictionary examples: finite forward/backward;
  Lorentz output constraint checked.
- Documented synthetic NC and LP train/evaluate workflows: completed in
  separate processes, without downloading data or accessing a cluster.
- Python wheel build: succeeded without fetching dependencies.
- Release packaging excludes benchmark data, weights, outputs, caches, and
  cluster launchers; file integrity is recorded in `MANIFEST.sha256`.

Runtime: Python 3.12.13, PyTorch 2.13.0, NumPy 2.5.1, SciPy 1.18.0,
scikit-learn 1.9.0, pytest 8.4.2. The original project license remains unlocated.

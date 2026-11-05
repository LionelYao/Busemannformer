# Reproducibility scope

The public-facing source is separated from the mutable research workspace.
The core attention implementation and clean LP model/split code are unchanged;
training runners are portable adaptations and have their own regression tests.
Original copied-file hashes appear in `SOURCE_PROVENANCE.json`. Release hashes
cover the final files, including the documented H200-test skip adjustment.

## What this package verifies

- Algebraic and gradient comparisons for attention implementations on tiny inputs.
- Positive boundary features, Lorentz outputs, sparse/dense LP score parity,
  and disjoint clean-link-prediction splits.
- Validation-selected training and a separate, provenance-checked test stage.
- Portable synthetic NC/LP round trips, without downloading benchmark data.

These checks are not evidence of reproduced benchmark accuracy or GPU speed.
The recorded local environment is Python 3.12.13, PyTorch 2.13.0, NumPy 2.5.1,
SciPy 1.18.0, scikit-learn 1.9.0, and pytest 8.4.2 on macOS CPU. Dependency ranges
are compatible-install constraints, not a complete historical lockfile. The
original H200 experiment environment has not been rerun for this release.

## Relation to the manuscript

| Result family | Included here | Still needed for exact table replay |
|---|---|---|
| Node classification | Model and generic sparse LD runner | Each selected dataset recipe, preprocessing, split provenance, and readout settings |
| Clean link prediction | Original model/split primitives and portable training | Frozen per-cell recipes and input provenance for each reported cohort |
| Improved Disease LP | Shared attention implementation | The dataset-specific validation-selection/confirmation protocol and its exact selected configuration |
| Large-scale experiments | Boundary-dictionary operator and its tests | Native Hypformer integration, baseline source versions, and all selected ogbn-arxiv recipes |
| Published baseline values | Not reimplemented or relabeled | Cite the original papers; distinguish references from reruns |

Do not use these example configurations to assert reproduction of 89.99%,
90.16%, or any other paper entry. There are no fabricated expected metrics.
Do not change a manuscript checklist answer to “complete reproducibility”
based on this package alone.

## Input and evaluation safeguards

Features must be prepared according to a declared dataset protocol. In LP,
numeric inputs can still contain leaked labels or full-graph structural
statistics: rejecting a `labels` array does not prove feature provenance is
clean. Recompute edge-derived features from the training graph only and record
the transformation. The generic runner cannot repair an already contaminated
feature matrix.

LP uses canonical undirected edges, disjoint positive partitions, and negatives
outside the complete known-positive set. Training negatives also exclude the
reserved evaluation negatives. Two-hop attention is built from training edges
only. NC uses the full graph, with disjoint node-label splits; this is a different,
transductive protocol.

Train all predeclared seeds, select settings using validation only, then evaluate
the frozen checkpoints. Report all seed outcomes with the metric and SD
convention explicitly stated. Separate commands and hashes prevent accidental
mix-ups; they do not make a previously inspected test set fresh or prevent
deliberate repeated tuning.

## Remaining release work

1. Locate and retain the original project license; confirm contributor and
   upstream attribution obligations. Do not substitute a third-party license.
2. Add the exact per-table configuration/data-provenance mapping if advertising
   this as a full paper-reproduction package.
3. Add the final paper citation and an archival DOI if one is issued.

Public source repository:
[LionelYao/Busemannformer](https://github.com/LionelYao/Busemannformer).
Public availability does not make this a complete paper-reproduction package.

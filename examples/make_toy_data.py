"""Create small synthetic NPZ fixtures, not replacements for paper datasets."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new directory; toy data is never overwritten")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(42)
    nodes = 48
    features = rng.uniform(0.1, 1, size=(nodes, 8)).astype(np.float32)
    features /= features.sum(axis=1, keepdims=True)
    edges = np.asarray([(i, (i + step) % nodes) for i in range(nodes) for step in (1, 3)], dtype=np.int64)
    permutation = rng.permutation(nodes)
    labels = np.arange(nodes, dtype=np.int64) % 2
    np.savez(args.output / "nc.npz", features=features, edges=edges, labels=labels,
             train_idx=permutation[:28], val_idx=permutation[28:38], test_idx=permutation[38:])
    np.savez(args.output / "lp.npz", features=features, edges=edges)
    print("Created nc.npz and lp.npz (48 synthetic nodes; no benchmark data)")


if __name__ == "__main__":
    main()

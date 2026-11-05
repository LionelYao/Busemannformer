"""Opt-in clean LP protocol, not a relabeling of historical results.

No graph loading, model execution, file writes or global RNG changes.
Implicit-complement sampling uses O(N + E + k), not O(N squared), memory.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

POLICY = "undirected-clean-nonedges-disjoint-evaluation-epoch-seeded-v1"


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def canonical_pairs(nodes, edges):
    """Union reversed/duplicate edges and drop source self-loops."""
    nodes = _integer(nodes, "nodes", 2)
    if nodes > 2**31:
        raise ValueError("Node count exceeds supported int64 pair encoding")
    pairs = np.asarray(edges)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("Edges must be an integer (E, 2) array")
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= nodes)):
        raise ValueError("Node index out of range")
    pairs = np.sort(pairs.astype(np.int64, copy=True), axis=1)
    return np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)


def pair_ranks(nodes, pairs):
    """Lexicographic index among N(N-1)/2 canonical u<v pairs."""
    u, v = pairs[:, 0], pairs[:, 1]
    return u * (2 * nodes - u - 1) // 2 + v - u - 1


def _sample_ranks(population, count, rng):
    """Floyd uniform subset sampling in O(count) memory, then random order."""
    population, count = _integer(population, "population"), _integer(count, "count")
    if count > population:
        raise ValueError("Not enough distinct eligible nonedges")
    chosen, ordered = set(), []
    for upper in range(population - count, population):
        draw = int(rng.integers(upper + 1))
        value = upper if draw in chosen else draw
        chosen.add(value)
        ordered.append(value)
    result = np.asarray(ordered, dtype=np.int64)
    rng.shuffle(result)
    return result


class NonedgeSampler:
    """Uniform off-diagonal pairs outside a fixed forbidden set."""

    def __init__(self, nodes, forbidden):
        self.nodes = _integer(nodes, "nodes", 2)
        self.forbidden = canonical_pairs(nodes, forbidden)
        self.forbidden_ranks = pair_ranks(self.nodes, self.forbidden)
        # Sorted b_i-i is nondecreasing. Complement rank r maps exactly to
        # full rank r + count(b_i-i <= r), without constructing the complement.
        self.complement_offsets = self.forbidden_ranks - np.arange(len(self.forbidden), dtype=np.int64)
        u = np.arange(self.nodes, dtype=np.int64)
        self.row_starts = u * (2 * self.nodes - u - 1) // 2
        self.available = self.nodes * (self.nodes - 1) // 2 - len(self.forbidden)
        for array in (self.forbidden, self.forbidden_ranks, self.complement_offsets, self.row_starts):
            array.setflags(write=False)

    def decode_complement_ranks(self, ranks):
        ranks = np.asarray(ranks)
        if ranks.ndim != 1 or ranks.dtype.kind not in "iu":
            raise ValueError("Complement ranks must be one-dimensional integers")
        if ranks.size and (np.any(ranks < 0) or np.any(ranks >= self.available)):
            raise ValueError("Complement rank out of range")
        ranks = ranks.astype(np.int64, copy=False)
        full = ranks + np.searchsorted(self.complement_offsets, ranks, side="right")
        u = np.searchsorted(self.row_starts, full, side="right") - 1
        v = full - self.row_starts[u] + u + 1
        return np.column_stack((u, v)).astype(np.int64, copy=False)

    def sample(self, count, rng):
        return self.decode_complement_ranks(_sample_ranks(self.available, count, rng))


def _array_hash(array):
    data = np.asarray(array, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(data.shape), separators=(",", ":")).encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CleanLPSplit:
    nodes: int
    seed: int
    train_positive: np.ndarray
    valid_positive: np.ndarray
    test_positive: np.ndarray
    valid_negative: np.ndarray
    test_negative: np.ndarray
    training_sampler: NonedgeSampler

    def training_negatives(self, epoch):
        """Same seed/epoch sequence across arms, independent of model RNG."""
        epoch = _integer(epoch, "epoch", 1)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1, epoch]))
        return self.training_sampler.sample(len(self.train_positive), rng)

    def provenance(self):
        arrays = {name: getattr(self, name) for name in (
            "train_positive", "valid_positive", "test_positive", "valid_negative", "test_negative")}
        payload = dict(policy=POLICY, nodes=self.nodes, seed=self.seed,
            arrays={name: dict(count=len(value), sha256=_array_hash(value)) for name, value in arrays.items()},
            training_eligible_nonedges=self.training_sampler.available,
            training_forbidden_sha256=_array_hash(self.training_sampler.forbidden),
            training_sampling="uniform_without_replacement_within_epoch",
            model_rng_used=False, dense_complement_materialized=False)
        payload["split_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return payload


def make_clean_lp_split(nodes, edges, seed=42, valid_fraction=0.05, test_fraction=0.10):
    seed, nodes = _integer(seed, "seed"), _integer(nodes, "nodes", 2)
    for value in (valid_fraction, test_fraction):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Evaluation fractions must be finite and strictly between zero and one")
    if valid_fraction + test_fraction >= 1:
        raise ValueError("Training fraction must be positive")
    positive = canonical_pairs(nodes, edges)
    n_valid, n_test = int(len(positive) * valid_fraction), int(len(positive) * test_fraction)
    if min(n_valid, n_test, len(positive) - n_valid - n_test) < 1:
        raise ValueError("Every positive split must contain at least one edge")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    shuffled = positive[rng.permutation(len(positive))]
    valid_pos, test_pos = shuffled[:n_valid].copy(), shuffled[n_valid:n_valid+n_test].copy()
    train_pos = shuffled[n_valid+n_test:].copy()
    evaluation_neg = NonedgeSampler(nodes, positive).sample(n_valid+n_test, rng)
    valid_neg, test_neg = evaluation_neg[:n_valid].copy(), evaluation_neg[n_valid:].copy()
    sampler = NonedgeSampler(nodes, np.concatenate((positive, evaluation_neg), axis=0))
    if sampler.available < len(train_pos):
        raise ValueError("Insufficient nonedges for unique epoch negatives after evaluation reservation")
    arrays = (train_pos, valid_pos, test_pos, valid_neg, test_neg)
    for array in arrays:
        array.setflags(write=False)
    return CleanLPSplit(nodes, seed, *arrays, sampler)

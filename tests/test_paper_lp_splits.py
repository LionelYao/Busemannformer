"""Tiny independent combinatorial checks, without benchmark data/models."""
from collections import Counter
from itertools import combinations, product

import numpy as np
import pytest

from paper_lp_splits import NonedgeSampler, _sample_ranks, canonical_pairs, make_clean_lp_split, pair_ranks


def pairs(array):
    return {tuple(map(int, row)) for row in array}


def graph():
    return np.array([(i, (i+1) % 20) for i in range(20)] + [(i, i+2) for i in range(10)], dtype=np.int64)


def test_exact_complement_bijection_for_every_tiny_forbidden_subset():
    universe = list(combinations(range(4), 2))
    for bits in product((0, 1), repeat=len(universe)):
        blocked = [edge for edge, bit in zip(universe, bits) if bit]
        sampler = NonedgeSampler(4, np.array(blocked, dtype=np.int64).reshape(-1, 2))
        decoded = sampler.decode_complement_ranks(np.arange(sampler.available))
        assert decoded.tolist() == [list(edge) for edge in universe if edge not in blocked]


def test_pair_ranks_match_independent_enumeration():
    for n in range(2, 18):
        edges = np.array(list(combinations(range(n), 2)), dtype=np.int64)
        np.testing.assert_array_equal(pair_ranks(n, edges), np.arange(len(edges)))


def test_floyd_all_tiny_draw_sequences_give_uniform_subsets():
    class FixedDraws:
        def __init__(self, draws):
            self.draws = iter(draws)
        def integers(self, bound):
            value = next(self.draws)
            assert 0 <= value < bound
            return value
        def shuffle(self, values):
            pass
    counts = Counter()
    for sequence in product(range(3), range(4)):
        sample = _sample_ranks(4, 2, FixedDraws(sequence))
        counts[tuple(sorted(sample.tolist()))] += 1
    assert set(counts) == set(combinations(range(4), 2))
    assert set(counts.values()) == {2}


def test_positive_evaluation_and_training_sets_are_clean_and_disjoint():
    split = make_clean_lp_split(20, graph(), valid_fraction=.2, test_fraction=.2)
    positives = [pairs(getattr(split, name)) for name in ("train_positive", "valid_positive", "test_positive")]
    all_positive = pairs(canonical_pairs(20, graph()))
    assert set.union(*positives) == all_positive
    assert all(not a & b for a, b in combinations(positives, 2))
    valid, test = pairs(split.valid_negative), pairs(split.test_negative)
    assert not valid & test and not (valid | test) & all_positive
    for epoch in range(1, 21):
        negative = split.training_negatives(epoch)
        assert len(pairs(negative)) == len(split.train_positive)
        assert all(u < v for u, v in negative)
        assert not pairs(negative) & (all_positive | valid | test)


def test_private_rng_and_epoch_pairing_independent_of_global_model_rng():
    np.random.seed(123)
    before = np.random.get_state()
    first = make_clean_lp_split(20, graph())
    negative = first.training_negatives(9)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.random.random(100)
    second = make_clean_lp_split(20, graph())
    assert first.provenance() == second.provenance()
    np.testing.assert_array_equal(negative, second.training_negatives(9))
    assert not np.array_equal(negative, first.training_negatives(10))
    assert first.provenance() != make_clean_lp_split(20, graph(), seed=43).provenance()


def test_duplicates_reverse_edges_and_source_loops_do_not_change_protocol():
    clean = graph()
    noisy = np.concatenate((clean, clean[:, ::-1], clean, np.array([[0, 0], [8, 8]])))
    assert make_clean_lp_split(20, clean).provenance() == make_clean_lp_split(20, noisy).provenance()


def test_split_arrays_and_sampler_inventory_are_read_only():
    split = make_clean_lp_split(20, graph())
    for array in (split.train_positive, split.valid_positive, split.test_positive,
                  split.valid_negative, split.test_negative, split.training_sampler.forbidden):
        with pytest.raises(ValueError):
            array[0] = 0


@pytest.mark.parametrize("edges", [np.array([[0, 20]]), np.array([[-1, 2]]),
    np.array([[0.0, 2.0]]), np.array([0, 2]), np.array([[False, True]])])
def test_invalid_edges_rejected(edges):
    with pytest.raises(ValueError):
        make_clean_lp_split(20, edges)


@pytest.mark.parametrize("value", [0, -1, 1, .95, float("nan"), True])
def test_invalid_evaluation_fraction_rejected(value):
    with pytest.raises(ValueError):
        make_clean_lp_split(20, graph(), valid_fraction=value)


def test_dense_graph_fails_instead_of_infinite_rejection_sampling():
    full = np.array(list(combinations(range(10), 2)), dtype=np.int64)
    with pytest.raises(ValueError, match="Not enough"):
        make_clean_lp_split(10, full)


@pytest.mark.parametrize("epoch", [0, -1, True, 1.5])
def test_invalid_epoch_rejected(epoch):
    with pytest.raises(ValueError):
        make_clean_lp_split(20, graph()).training_negatives(epoch)


def test_empty_and_full_eligible_samples():
    sampler = NonedgeSampler(6, np.array([[0, 1], [0, 2], [3, 4]], dtype=np.int64))
    assert sampler.sample(0, np.random.default_rng(1)).shape == (0, 2)
    expected = set(combinations(range(6), 2)) - {(0, 1), (0, 2), (3, 4)}
    assert pairs(sampler.sample(sampler.available, np.random.default_rng(1))) == expected
    with pytest.raises(ValueError):
        sampler.sample(sampler.available+1, np.random.default_rng(1))

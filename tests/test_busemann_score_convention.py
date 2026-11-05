"""Scalar regression for the raw-log versus normalized Busemann convention.

These tests derive ideal unclamped scores independently with standard-library
math. They do not import production models, load data, or certify floating-point
kernel parity. Curvature is -c**2 and geometric strength is a(x)=c*tau*g(x).
"""

import math

import pytest


QUERY_SPATIAL = ((0.2, 0.3), (2.0, 0.7))
KEY_SPATIAL = ((0.4, -0.2), (2.0, 1.0), (-5.0, 0.5))


def _point(spatial, c):
    return (math.hypot(1.0 / c, *spatial), *spatial)


def _direction(point):
    length = math.hypot(*point[1:])
    assert length > 0
    return tuple(value / length for value in point[1:])


def _gap(query, key):
    return key[0] - math.fsum(a * b for a, b in zip(_direction(query), key[1:]))


def _gain(point):
    # The production LD radius proxy; not a curvature-invariant depth.
    return 1.0 + 0.7 * math.log(max(point[0], 1.0 + 1e-6))


def _directed(query, key, c, tau, gain):
    gap = _gap(query, key)
    assert gap > 1e-6  # All cases are strictly outside the production gap floor.
    raw = -tau * gain * math.log(gap)
    busemann = math.log(c * gap) / c
    geometric = -(c * tau * gain) * busemann
    return raw, geometric


def _bidirectional(query, key, c, tau, query_gain, key_gain):
    forward = _directed(query, key, c, tau, query_gain)
    reverse = _directed(key, query, c, tau, key_gain)
    return tuple(0.5 * (left + right) for left, right in zip(forward, reverse))


def _softmax(scores):
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _points(c):
    return ([_point(value, c) for value in QUERY_SPATIAL],
            [_point(value, c) for value in KEY_SPATIAL])


@pytest.mark.parametrize("c", (0.4, 1.0, 2.0))
def test_inputs_are_lorentz_points_with_unit_boundary_directions(c):
    queries, keys = _points(c)
    for point in queries + keys:
        assert point[0] > math.hypot(*point[1:])
        assert -point[0] ** 2 + math.fsum(x * x for x in point[1:]) == pytest.approx(
            -1.0 / c**2, abs=1e-13)
        assert math.hypot(*_direction(point)) == pytest.approx(1.0, abs=1e-15)
    assert all(_gap(q, k) > 1e-6 and _gap(k, q) > 1e-6
               for q in queries for k in keys)


@pytest.mark.parametrize("c", (0.4, 1.0, 2.0))
@pytest.mark.parametrize("tau", (0.3, 2.0))
def test_directed_offset_is_query_row_constant_and_softmax_cancels(c, tau):
    queries, keys = _points(c)
    for query in queries:
        gain = _gain(query)
        pairs = [_directed(query, key, c, tau, gain) for key in keys]
        expected_offset = tau * gain * math.log(c)
        for raw, geometric in pairs:
            assert raw - geometric == pytest.approx(expected_offset, abs=1e-13)
        assert _softmax([pair[0] for pair in pairs]) == pytest.approx(
            _softmax([pair[1] for pair in pairs]), rel=1e-13, abs=1e-14)


@pytest.mark.parametrize("c", (0.4, 1.0, 2.0))
@pytest.mark.parametrize("tau", (0.3, 2.0))
def test_bidirectional_offset_has_query_and_key_gain(c, tau):
    queries, keys = _points(c)
    assert len(set(_gain(key) for key in keys)) == len(keys)
    for query in queries:
        for key in keys:
            query_gain, key_gain = _gain(query), _gain(key)
            raw, geometric = _bidirectional(query, key, c, tau, query_gain, key_gain)
            expected = 0.5 * tau * (query_gain + key_gain) * math.log(c)
            assert raw - geometric == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("tau", (0.3, 2.0))
def test_unit_curvature_bidirectional_scores_and_softmax_coincide(tau):
    c = 1.0
    queries, keys = _points(c)
    for query in queries:
        pairs = [_bidirectional(query, key, c, tau, _gain(query), _gain(key))
                 for key in keys]
        assert [pair[0] for pair in pairs] == [pair[1] for pair in pairs]
        assert _softmax([pair[0] for pair in pairs]) == _softmax([pair[1] for pair in pairs])


@pytest.mark.parametrize("c", (0.4, 2.0))
@pytest.mark.parametrize("tau", (0.3, 2.0))
def test_nonunit_curvature_varying_key_gain_changes_bidirectional_softmax(c, tau):
    queries, keys = _points(c)
    for query in queries:
        pairs = [_bidirectional(query, key, c, tau, _gain(query), _gain(key))
                 for key in keys]
        raw_weights = _softmax([pair[0] for pair in pairs])
        geometric_weights = _softmax([pair[1] for pair in pairs])
        assert max(abs(a - b) for a, b in zip(raw_weights, geometric_weights)) > 1e-5
        # Softmax log odds equal score differences. The query term cancels,
        # but the key term remains; it cannot be removed by a query-row shift.
        for index in (1, 2):
            log_odds_change = (
                math.log(raw_weights[index] / raw_weights[0])
                - math.log(geometric_weights[index] / geometric_weights[0]))
            expected = 0.5 * tau * (_gain(keys[index]) - _gain(keys[0])) * math.log(c)
            assert log_odds_change == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("c", (0.4, 1.0, 2.0))
@pytest.mark.parametrize("tau", (0.3, 2.0))
def test_constant_key_gain_restores_bidirectional_softmax_equivalence(c, tau):
    queries, keys = _points(c)
    key_gain = 1.7
    for query in queries:
        # Query gains may differ across rows; only key gain must be constant
        # within each row for the remaining offset to be a row constant.
        pairs = [_bidirectional(query, key, c, tau, _gain(query), key_gain)
                 for key in keys]
        assert _softmax([pair[0] for pair in pairs]) == pytest.approx(
            _softmax([pair[1] for pair in pairs]), rel=1e-13, abs=1e-14)

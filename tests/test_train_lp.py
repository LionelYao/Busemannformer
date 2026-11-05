"""Small synthetic checks: no downloads, cluster access or paper-data fitting."""
import json

import numpy as np
import pytest
import torch

import train_lp as runner


@pytest.fixture
def fixture(tmp_path):
    nodes = 24
    edges = np.asarray([(i, (i + step) % nodes) for i in range(nodes) for step in (1, 2)])
    features = np.random.default_rng(12).normal(0, .1, size=(nodes, 4)).astype(np.float32)
    path = tmp_path / "graph.npz"
    np.savez(path, features=features, edges=edges)
    config = dict(width=8, heads=2, layers=1, dropout=0., epochs=2, patience=2, seed=17)
    return path, config, tmp_path / "run"


def test_train_then_explicit_evaluate_and_refuse_overwrite(fixture, tmp_path):
    path, config, run = fixture
    manifest = runner.train(path, run, config)
    assert manifest["record"]["test_scored"] is False
    environment = manifest["record"]["environment"]
    assert {"python", "torch", "numpy", "scipy", "scikit_learn", "cuda", "gpu",
            "deterministic_algorithms", "deterministic_algorithms_warn_only"} <= set(environment)
    assert environment["device"] == "cpu" and environment["gpu"] is None
    assert "test" not in manifest["record"]
    assert len(manifest["record"]["history"]) == 2
    with pytest.raises(FileExistsError):
        runner.train(path, run, config)
    result_path = tmp_path / "test.json"
    result = runner.evaluate(path, run, result_path)
    assert 0 <= result["test"]["roc_auc"] <= 1
    assert 0 <= result["test"]["average_precision"] <= 1
    assert result["seed"] == 17
    with pytest.raises(FileExistsError):
        runner.evaluate(path, run, result_path)
    assert (run / "test_evaluation_started.json").is_file()


def test_alternate_output_cannot_repeat_test_scoring(fixture, tmp_path, monkeypatch):
    path, config, run = fixture
    runner.train(path, run, config)
    runner.evaluate(path, run, tmp_path / "first.json")
    monkeypatch.setattr(runner, "edge_scores", lambda *args: pytest.fail("Repeated test scoring"))
    with pytest.raises(FileExistsError, match="already started"):
        runner.evaluate(path, run, tmp_path / "second.json")
    assert not (tmp_path / "second.json").exists()


@pytest.mark.parametrize("name", ["manifest.json", "checkpoint.pt", "test_evaluation_started.json"])
def test_test_output_cannot_replace_frozen_run_artifacts(fixture, monkeypatch, name):
    path, config, run = fixture
    runner.train(path, run, config)
    monkeypatch.setattr(runner, "edge_scores", lambda *args: pytest.fail("Scored test for protected output"))
    with pytest.raises((FileExistsError, ValueError)):
        runner.evaluate(path, run, run / name)
    assert not (run / "test_evaluation_started.json").exists()


def test_training_never_scores_heldout_edges(fixture, monkeypatch):
    path, config, run = fixture
    cfg = runner.validate_config(config)
    _, _, split, _ = runner.prepare(path, cfg, "cpu")
    original = runner.edge_scores
    calls = []
    def validation_only(model, x, support, positive, negative):
        np.testing.assert_array_equal(positive.cpu().numpy(), split.valid_positive)
        np.testing.assert_array_equal(negative.cpu().numpy(), split.valid_negative)
        calls.append(1)
        return original(model, x, support, positive, negative)
    monkeypatch.setattr(runner, "edge_scores", validation_only)
    runner.train(path, run, config)
    assert len(calls) == 3  # Two epochs plus selected validation replay.


def test_support_uses_train_graph_only(fixture):
    path, config, _ = fixture
    _, support, split, _ = runner.prepare(path, runner.validate_config(config), "cpu")
    adj = np.eye(split.nodes, dtype=bool)
    u, v = split.train_positive.T
    adj[u, v] = adj[v, u] = True
    expected = (adj.astype(np.int64) @ adj.astype(np.int64)) > 0
    observed = np.zeros_like(expected)
    observed[tuple(support.numpy())] = True
    np.testing.assert_array_equal(observed, expected)
    reserved = {tuple(x) for values in (split.valid_negative, split.test_negative) for x in values}
    for epoch in (1, 2, 3):
        assert not reserved.intersection(map(tuple, split.training_negatives(epoch)))
    assert not np.array_equal(split.training_negatives(1), split.training_negatives(2))


@pytest.mark.parametrize("change", ["input", "config", "source", "split", "checkpoint"])
def test_changed_provenance_rejected_before_test_scoring(fixture, tmp_path, monkeypatch, change):
    path, config, run = fixture
    runner.train(path, run, config)
    supplied = None
    if change == "input":
        with np.load(path) as archive:
            features, edges = archive["features"], archive["edges"]
        np.savez(path, features=features + .01, edges=edges)
    elif change == "config":
        supplied = config | {"lr": .01}
    elif change in ("source", "split"):
        manifest = runner.read_json(run / "manifest.json")
        if change == "source":
            manifest["record"]["sources"]["../unexpected.py"] = {"sha256": "0" * 64, "bytes": 1}
        else:
            manifest["record"]["provenance"]["split"]["seed"] += 1
        (run / "manifest.json").write_text(json.dumps(manifest))
    else:
        with (run / "checkpoint.pt").open("ab") as stream:
            stream.write(b"changed")
    monkeypatch.setattr(runner, "edge_scores", lambda *a: pytest.fail("Scored test despite changed provenance"))
    with pytest.raises(ValueError):
        runner.evaluate(path, run, tmp_path / "test.json", config=supplied)
    assert not (tmp_path / "test.json").exists()
    assert not (run / "test_evaluation_started.json").exists()


def test_unexpected_npz_keys_and_nonfinite_features_rejected(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, features=np.ones((4, 2)), edges=np.array([[0, 1]]), labels=np.zeros(4))
    with pytest.raises(ValueError, match="exactly features and edges"):
        runner.load_arrays(path)
    np.savez(path, features=np.full((4, 2), np.nan), edges=np.array([[0, 1]]))
    with pytest.raises(ValueError, match="finite"):
        runner.load_arrays(path)


def test_features_are_not_normalized(fixture):
    path, config, _ = fixture
    with np.load(path) as archive:
        original = archive["features"]
    features, _, _, _ = runner.prepare(path, runner.validate_config(config), "cpu")
    np.testing.assert_array_equal(features.numpy(), original)


def test_metrics_rank_logits_without_sigmoid_ties():
    assert runner.rank_metrics(torch.tensor([100., 90., 80., 70.]), 2)["roc_auc"] == 1.


@pytest.mark.parametrize("config", [{"unknown": 1}, {"seed": -1}, {"epochs": 0},
    {"heads": 3}, {"lr": float("nan")}, {"dropout": 1.}, {"valid_fraction": .95}])
def test_invalid_config_rejected(config):
    with pytest.raises(ValueError):
        runner.validate_config(config)

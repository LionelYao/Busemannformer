"""Synthetic smoke checks only; no real benchmark data or claimed paper scores."""

import json
from contextlib import contextmanager

import numpy as np
import pytest
import torch

import train_nc as runner


def toy_data():
    rng = np.random.default_rng(7)
    return {
        "features": rng.normal(size=(12, 3)).astype(np.float32),
        "edges": np.array([[i, i + 1] for i in range(11)], dtype=np.int64),
        "labels": np.arange(12, dtype=np.int64) % 2,
        "train_idx": np.arange(6, dtype=np.int64),
        "val_idx": np.arange(6, 10, dtype=np.int64),
        "test_idx": np.arange(10, 12, dtype=np.int64),
    }


def tiny_config():
    return {"d_model": 8, "num_heads": 2, "num_layers": 1, "max_epochs": 2, "patience": 2, "dropout": 0.0}


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_roundtrip_freezes_validation_and_evaluates_once(tmp_path):
    source = tmp_path / "input.npz"
    np.savez(source, **toy_data())
    run = tmp_path / "run"
    manifest = runner.train(source, run, tiny_config(), seed=3)
    assert manifest["model_protocol"]["use_gnn_branch"] is False
    assert manifest["best_epoch"] in (1, 2)
    assert manifest["config"]["nc_head"] == "hypcls"
    assert manifest["recipe_status"] == "portable_example_not_paper_table_reproduction"
    assert manifest["train_class_weights"] == [1.0, 1.0]
    assert not (run / "test_metrics.json").exists()
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2
    assert all(set(row) == {"epoch", "train_loss", "val_macro_f1"} for row in history)
    checkpoint = torch.load(run / "best.pt", weights_only=True)
    assert checkpoint["metadata"]["best_epoch"] == manifest["best_epoch"]
    result = runner.evaluate(source, run)
    assert 0 <= result["test_accuracy"] <= 1
    assert 0 <= result["test_macro_f1"] <= 1
    assert result["num_test_nodes"] == 2
    with pytest.raises(FileExistsError):
        runner.evaluate(source, run)
    with pytest.raises(FileExistsError):
        runner.evaluate(source, run, output=tmp_path / "second.json")
    with pytest.raises(FileExistsError):
        runner.train(source, run, tiny_config(), seed=3)


@pytest.mark.parametrize("change", [
    {"test_idx": np.array([9, 11])},
    {"test_idx": np.array([11])},
    {"test_idx": np.array([10, 12])},
    {"val_idx": np.array([6, 7, 8, 8])},
    {"train_idx": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])},
    {"edges": np.array([[0, 12]])},
    {"features": np.full((12, 3), np.nan)},
    {"labels": np.full(12, 9)},
])
def test_invalid_npz_is_rejected(tmp_path, change):
    data = toy_data()
    data.update(change)
    source = tmp_path / "invalid.npz"
    np.savez(source, **data)
    with pytest.raises(ValueError):
        runner.load_data(source)


def test_object_arrays_are_rejected(tmp_path):
    data = toy_data()
    data["features"] = np.array([[{"unsafe": 1}]], dtype=object)
    source = tmp_path / "object.npz"
    np.savez(source, **data)
    with pytest.raises(ValueError):
        runner.load_data(source)


def test_two_hop_support_matches_dense_reference():
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    support = runner.two_hop_support(edges, 4).numpy()
    actual = set(map(tuple, support.T))
    assert actual == {(i, j) for i in range(3) for j in range(3)} | {(3, 3)}


def test_sparse_model_matches_dense_mask():
    data = toy_data()
    config = runner.resolve_config(tiny_config())
    torch.manual_seed(7)
    model = runner.build_model(data, config).eval()
    support = runner.two_hop_support(data["edges"], 12)
    mask = torch.ones(1, 12, 12, dtype=torch.bool)
    mask[0, support[0], support[1]] = False
    features = torch.from_numpy(data["features"])
    with torch.no_grad():
        sparse = model(features, attn_edge_index=support)
        dense = model(features, bool_mask=mask)
    torch.testing.assert_close(sparse, dense, atol=2e-5, rtol=2e-5)


def test_artifact_and_input_tampering_is_rejected(tmp_path):
    source = tmp_path / "input.npz"
    np.savez(source, **toy_data())
    run = tmp_path / "run"
    runner.train(source, run, tiny_config(), seed=9)
    changed = toy_data()
    changed["features"][0, 0] += 1
    other = tmp_path / "changed.npz"
    np.savez(other, **changed)
    with pytest.raises(ValueError, match="Input hash mismatch"):
        runner.evaluate(other, run)
    assert not (run / "test_evaluation_started.json").exists()
    (run / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        runner.evaluate(source, run)


def test_test_labels_do_not_affect_training_checkpoint(tmp_path):
    data = toy_data()
    one = tmp_path / "one.npz"
    two = tmp_path / "two.npz"
    np.savez(one, **data)
    data["labels"][data["test_idx"]] = 1 - data["labels"][data["test_idx"]]
    np.savez(two, **data)
    run_one, run_two = tmp_path / "run_one", tmp_path / "run_two"
    runner.train(one, run_one, tiny_config(), seed=13)
    runner.train(two, run_two, tiny_config(), seed=13)
    first = torch.load(run_one / "best.pt", weights_only=True)["state_dict"]
    second = torch.load(run_two / "best.pt", weights_only=True)["state_dict"]
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    assert (run_one / "history.json").read_text() == (run_two / "history.json").read_text()


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown configuration"):
        runner.resolve_config({"epochs_typo": 3})


def test_environment_overrides_are_rejected_before_io(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSFORMER_SPARSE_QUERY_CHUNK", "1")
    with pytest.raises(ValueError, match="environment overrides"):
        runner.train(tmp_path / "missing.npz", tmp_path / "run")
    with pytest.raises(ValueError, match="environment overrides"):
        runner.evaluate(tmp_path / "missing.npz", tmp_path / "run")


def test_input_change_during_load_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    np.savez(source, **toy_data())
    original = runner.np.load

    @contextmanager
    def changing_load(*args, **kwargs):
        with original(*args, **kwargs) as archive:
            yield archive
        with source.open("ab") as handle:
            handle.write(b"changed")

    monkeypatch.setattr(runner.np, "load", changing_load)
    with pytest.raises(ValueError, match="changed while being read"):
        runner.load_data(source)


def test_input_change_during_training_preparation_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    np.savez(source, **toy_data())
    original = runner.load_data

    def changing_load(path):
        data = original(path)
        with source.open("ab") as handle:
            handle.write(b"changed")
        return data

    monkeypatch.setattr(runner, "load_data", changing_load)
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="preparing training"):
        runner.train(source, run, tiny_config())
    assert not run.exists()


@pytest.mark.parametrize("change", ["input", "source"])
def test_change_during_fitting_prevents_checkpoint_save(tmp_path, monkeypatch, change):
    source = tmp_path / "input.npz"
    np.savez(source, **toy_data())
    run = tmp_path / "run"
    original = runner.f1_score

    def changing_score(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "input":
            with source.open("ab") as handle:
                handle.write(b"changed")
        else:
            monkeypatch.setattr(runner, "source_hashes", lambda: {"changed": "changed"})
        return result

    monkeypatch.setattr(runner, "f1_score", changing_score)
    with pytest.raises(ValueError, match="during fitting"):
        runner.train(source, run, tiny_config())
    assert not (run / "best.pt").exists()
    assert not (run / "manifest.json").exists()


def test_input_change_during_evaluation_preparation_prevents_scoring(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    np.savez(source, **toy_data())
    run = tmp_path / "run"
    runner.train(source, run, tiny_config())
    original = runner.load_data

    def changing_load(path):
        data = original(path)
        with source.open("ab") as handle:
            handle.write(b"changed")
        return data

    monkeypatch.setattr(runner, "load_data", changing_load)
    with pytest.raises(ValueError, match="preparing evaluation"):
        runner.evaluate(source, run)
    assert not (run / "test_evaluation_started.json").exists()
    assert not (run / "test_metrics.json").exists()


def test_input_change_during_test_evaluation_prevents_result_save(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    np.savez(source, **toy_data())
    run = tmp_path / "run"
    runner.train(source, run, tiny_config())
    original = runner.accuracy_score

    def changing_score(*args, **kwargs):
        result = original(*args, **kwargs)
        with source.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(runner, "accuracy_score", changing_score)
    with pytest.raises(ValueError, match="during evaluation"):
        runner.evaluate(source, run)
    assert (run / "test_evaluation_started.json").exists()
    assert not (run / "test_metrics.json").exists()

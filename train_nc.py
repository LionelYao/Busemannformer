"""Portable sparse node-classification example, not a paper-table recipe.

Input is an allow_pickle=False NPZ with preprocessed floating-point features
[N,F], integer edges [E,2], integer labels [N], and disjoint, exhaustive integer
train_idx / val_idx / test_idx vectors. Graph structure and features are shared
transductively. Only train labels enter the loss or class weights; only validation
macro-F1 selects the checkpoint. A separate command evaluates the frozen test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.nn import functional as F

from busemann_transformer import BusGraphTransformerNC


DEFAULT_CONFIG = {
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "nc_head": "hypcls",
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "max_epochs": 500,
    "patience": 80,
    "grad_clip": 1.0,
}
MODEL_PROTOCOL = {
    "score_type": "busemann_log_depth",
    "attn_proj": "hyp_linear",
    "curvature": 1.0,
    "learn_curvature": False,
    "final_readout_curvature": "unit",
    "use_gnn_branch": False,
    "support": "undirected_two_hop_plus_self",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_hashes():
    here = Path(__file__).resolve().parent
    return {
        name: sha256_file(here / name)
        for name in ("train_nc.py", "busemann_transformer.py")
    }


def resolve_config(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(value) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **value}
    for name in ("d_model", "num_heads", "num_layers", "max_epochs", "patience"):
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if result["d_model"] % result["num_heads"]:
        raise ValueError("d_model must be divisible by num_heads")
    for name in ("dropout", "learning_rate", "weight_decay", "grad_clip"):
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= result["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    if result["learning_rate"] <= 0 or result["grad_clip"] <= 0:
        raise ValueError("learning_rate and grad_clip must be positive")
    if result["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    if result["nc_head"] not in ("hypcls", "bmlr", "tangent_linear"):
        raise ValueError("nc_head must be hypcls, bmlr, or tangent_linear")
    return result


def load_data(path):
    before = sha256_file(path)
    required = {"features", "edges", "labels", "train_idx", "val_idx", "test_idx"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ keys: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
    features = arrays["features"]
    if features.ndim != 2 or min(features.shape) < 1 or features.dtype.kind != "f":
        raise ValueError("features must be a nonempty floating-point [N,F] matrix")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    features = features.astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("features overflow float32")
    n = features.shape[0]
    labels = arrays["labels"]
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError("labels must be an integer vector of length N")
    if np.any(labels < 0) or np.any(labels >= n):
        raise ValueError("labels must be nonnegative class IDs smaller than N")
    edges = arrays["edges"]
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("edges must be an integer [E,2] matrix")
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("edges contain out-of-range node IDs")
    indices = {}
    for name in ("train_idx", "val_idx", "test_idx"):
        values = arrays[name]
        if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a nonempty integer vector")
        if np.any(values < 0) or np.any(values >= n):
            raise ValueError(f"{name} contains out-of-range node IDs")
        indices[name] = values.astype(np.int64)
    merged = np.concatenate(list(indices.values()))
    if len(merged) != n or not np.array_equal(np.sort(merged), np.arange(n)):
        raise ValueError("Splits must be disjoint, duplicate-free, and exhaustive")
    train_classes = np.unique(labels[indices["train_idx"]])
    if len(train_classes) < 2 or not np.array_equal(train_classes, np.arange(len(train_classes))):
        raise ValueError("Training labels must cover contiguous class IDs 0,...,C-1, C>=2")
    if np.any(labels >= len(train_classes)):
        raise ValueError("All classes must be represented in training")
    if sha256_file(path) != before:
        raise ValueError("Input changed while being read")
    return {
        "features": features,
        "edges": edges.astype(np.int64),
        "labels": labels.astype(np.int64),
        **indices,
    }


def split_hashes(data):
    return {
        name: hashlib.sha256(data[name].astype("<i8", copy=False).tobytes()).hexdigest()
        for name in ("train_idx", "val_idx", "test_idx")
    }


def two_hop_support(edges, num_nodes):
    """Return [2,E_support] row=query edges without allocating dense adjacency.

    Two-hop support can itself approach N squared on dense/high-degree graphs;
    sparse execution is not a guarantee of linear complexity on arbitrary graphs.
    """
    adjacency = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.bool_), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T).maximum(sp.eye(num_nodes, dtype=np.bool_))
    support = (adjacency @ adjacency).tocsr()
    support.eliminate_zeros()
    support.sort_indices()
    row, col = support.nonzero()
    return torch.from_numpy(np.stack((row, col)).astype(np.int64))


def build_model(data, config):
    return BusGraphTransformerNC(
        d_in=data["features"].shape[1],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=int(data["labels"][data["train_idx"]].max()) + 1,
        c=1.0,
        dropout=config["dropout"],
        learn_curvature=False,
        score_type="busemann_log_depth",
        attn_proj="hyp_linear",
        head_type=config["nc_head"],
        final_readout_curvature="unit",
        use_attn_bias=False,
        use_gnn_branch=False,
    )


def _device(name):
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(name)


def _reject_environment_overrides():
    names = sorted(name for name in os.environ if name.startswith("BUSFORMER_"))
    if names:
        raise ValueError(
            "Portable NC rejects undocumented BUSFORMER_* environment overrides; "
            "unset these variables: " + ", ".join(names)
        )


def _seed_everything(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _runtime(device):
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms_warn_only": True,
    }


def train(input_path, output, config=None, seed=42, device="cpu"):
    """Fit and select only on validation; never compute test-set metrics."""
    _reject_environment_overrides()
    config = resolve_config(config)
    device = _device(device)
    _seed_everything(seed)
    input_hash = sha256_file(input_path)
    sources = source_hashes()
    data = load_data(input_path)
    if sha256_file(input_path) != input_hash or source_hashes() != sources:
        raise ValueError("Input or source changed while preparing training")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config)
    features = torch.from_numpy(data["features"]).to(device)
    labels = torch.from_numpy(data["labels"]).to(device)
    train_idx = torch.from_numpy(data["train_idx"]).to(device)
    support = two_hop_support(data["edges"], len(features)).to(device)
    model = build_model(data, config).to(device)
    num_classes = int(labels[train_idx].max().item()) + 1
    counts = torch.bincount(labels[train_idx], minlength=num_classes).float()
    weights = len(train_idx) / (num_classes * counts)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_score, best_epoch, stale, best_state = -1.0, -1, 0, None
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, attn_edge_index=support)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(features, attn_edge_index=support)[data["val_idx"]]
            if not torch.isfinite(val_logits).all():
                raise FloatingPointError(f"Non-finite validation logits at epoch {epoch}")
            prediction = val_logits.argmax(dim=-1).cpu().numpy()
        score = float(f1_score(
            data["labels"][data["val_idx"]], prediction,
            labels=np.arange(num_classes), average="macro", zero_division=0,
        ))
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_macro_f1": score})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if sha256_file(input_path) != input_hash or source_hashes() != sources:
        raise ValueError("Input or source changed during fitting")
    metadata = {
        "format_version": 1,
        "task": "transductive_node_classification",
        "recipe_status": "portable_example_not_paper_table_reproduction",
        "config": config,
        "config_sha256": _json_digest(config),
        "model_protocol": MODEL_PROTOCOL,
        "seed": seed,
        "input_sha256": input_hash,
        "split_sha256": split_hashes(data),
        "source_sha256": sources,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "num_nodes": len(features),
        "num_features": features.shape[1],
        "num_classes": num_classes,
        "support_edges": support.shape[1],
        "train_class_weights": weights.cpu().tolist(),
        "runtime": _runtime(device),
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, output / "best.pt")
    _write_json(output / "history.json", history)
    manifest = {
        **metadata,
        "artifacts_sha256": {
            name: sha256_file(output / name)
            for name in ("config.json", "history.json", "best.pt")
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def evaluate(input_path, run, device="cpu", output=None):
    """Verify frozen artifacts and score this run's test split once.

    Hash checks detect accidental mismatches, not deliberate alteration of all
    files and their hashes. The exclusive marker prevents repeated evaluation by
    this command; it is an audit safeguard, not a security boundary.
    """
    _reject_environment_overrides()
    device = _device(device)
    run = Path(run)
    output = run / "test_metrics.json" if output is None else Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected_artifacts = {"config.json", "history.json", "best.pt"}
    if set(manifest["artifacts_sha256"]) != expected_artifacts:
        raise ValueError("Unexpected artifact manifest")
    for name in expected_artifacts:
        if sha256_file(run / name) != manifest["artifacts_sha256"][name]:
            raise ValueError(f"Artifact hash mismatch: {name}")
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("Source hash mismatch; evaluate with the training source version")
    if manifest["input_sha256"] != sha256_file(input_path):
        raise ValueError("Input hash mismatch")
    config = resolve_config(json.loads((run / "config.json").read_text(encoding="utf-8")))
    if config != manifest["config"] or _json_digest(config) != manifest["config_sha256"]:
        raise ValueError("Configuration mismatch")
    if manifest["model_protocol"] != MODEL_PROTOCOL or manifest["format_version"] != 1:
        raise ValueError("Unsupported model protocol or artifact version")
    data = load_data(input_path)
    if manifest["split_sha256"] != split_hashes(data):
        raise ValueError("Split fingerprint mismatch")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    expected_metadata = {key: value for key, value in manifest.items() if key != "artifacts_sha256"}
    if checkpoint["metadata"] != expected_metadata:
        raise ValueError("Checkpoint metadata mismatch")
    _seed_everything(manifest["seed"])
    model = build_model(data, config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    features = torch.from_numpy(data["features"]).to(device)
    support = two_hop_support(data["edges"], len(features)).to(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    if sha256_file(input_path) != manifest["input_sha256"] or source_hashes() != manifest["source_sha256"]:
        raise ValueError("Input or source changed while preparing evaluation")
    # A failed evaluation leaves the marker for inspection; do not silently retry.
    _write_json(run / "test_evaluation_started.json", {
        "checkpoint_sha256": manifest["artifacts_sha256"]["best.pt"],
        "input_sha256": manifest["input_sha256"],
        "result_filename": output.name,
    })
    with torch.no_grad():
        logits = model(features, attn_edge_index=support)[data["test_idx"]]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite test logits")
        prediction = logits.argmax(dim=-1).cpu().numpy()
    truth = data["labels"][data["test_idx"]]
    result = {
        "task": manifest["task"],
        "recipe_status": manifest["recipe_status"],
        "seed": manifest["seed"],
        "num_test_nodes": len(truth),
        "test_accuracy": float(accuracy_score(truth, prediction)),
        "test_macro_f1": float(f1_score(
            truth, prediction, labels=np.arange(manifest["num_classes"]),
            average="macro", zero_division=0,
        )),
        "metric_scale": "fraction_0_to_1",
        "checkpoint_sha256": manifest["artifacts_sha256"]["best.pt"],
        "input_sha256": manifest["input_sha256"],
        "config_sha256": manifest["config_sha256"],
        "split_sha256": manifest["split_sha256"],
        "source_sha256": manifest["source_sha256"],
        "runtime": _runtime(device),
    }
    if sha256_file(input_path) != manifest["input_sha256"] or source_hashes() != manifest["source_sha256"]:
        raise ValueError("Input or source changed during evaluation")
    _write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Train/validation only; no test metrics")
    fit.add_argument("--input", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path, help="New run directory")
    fit.add_argument("--config", required=True, type=Path, help="JSON configuration; unspecified keys use defaults")
    fit.add_argument("--seed", default=42, type=int)
    fit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    score = commands.add_parser("evaluate", help="Verify and evaluate a frozen run once")
    score.add_argument("--input", required=True, type=Path)
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--output", type=Path, help="Default: RUN/test_metrics.json")
    score.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = train(args.input, args.output, config, args.seed, args.device)
        print(json.dumps({"best_epoch": result["best_epoch"], "best_val_macro_f1": result["best_val_macro_f1"]}))
    else:
        print(json.dumps(evaluate(args.input, args.run, args.device, args.output), sort_keys=True))


if __name__ == "__main__":
    main()

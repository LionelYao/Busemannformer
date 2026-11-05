"""Portable clean link prediction; not an exact historical paper replay.

Inputs are trusted, already-preprocessed numeric NPZ arrays ``features`` (N,F)
and ``edges`` (E,2). No node labels, downloads, or cluster APIs are used.
Training selects only validation AUC. Test scoring requires a separate command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F

from paper_lp_model import ARMS, build_model
from paper_lp_splits import make_clean_lp_split


SCHEMA = "busemannformer-portable-lp-v1"
SOURCE_FILES = ("train_lp.py", "busemann_transformer.py", "paper_lp_model.py", "paper_lp_splits.py")
DEFAULT_CONFIG = dict(arm="directed_p2", seed=42, width=64, heads=4,
    layers=2, dropout=0.1, lr=5e-4, weight_decay=1e-4,
    epochs=500, patience=80, valid_fraction=0.05, test_fraction=0.10)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("A regular non-symlink input file is required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def source_fingerprints():
    root = Path(__file__).resolve().parent
    return {name: fingerprint(root / name) for name in SOURCE_FILES}


def runtime_metadata(device):
    """Record actual runtime flags; do not promise cross-device determinism."""
    device = torch.device(device)
    return dict(python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__,
        device=str(device), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_algorithms_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_config(supplied=None):
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown configuration fields")
    config = DEFAULT_CONFIG | supplied
    if config["arm"] not in ARMS:
        raise ValueError("Unknown attention/decoder arm")
    for name in ("seed", "width", "heads", "layers", "epochs", "patience"):
        if type(config[name]) is not int or config[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid integer configuration: {name}")
    if config["width"] % config["heads"]:
        raise ValueError("Width must be divisible by the number of heads")
    for name in ("dropout", "lr", "weight_decay", "valid_fraction", "test_fraction"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid numeric configuration: {name}")
    if not 0 <= config["dropout"] < 1 or config["lr"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid dropout or optimizer settings")
    if not (0 < config["valid_fraction"] < 1 and 0 < config["test_fraction"] < 1
            and config["valid_fraction"] + config["test_fraction"] < 1):
        raise ValueError("Invalid evaluation fractions")
    return config


def load_arrays(path):
    before = fingerprint(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"features", "edges"}:
            raise ValueError("NPZ must contain exactly features and edges; node labels are not accepted")
        features, edges = archive["features"], archive["edges"]
    if features.ndim != 2 or min(features.shape) < 1 or features.dtype.kind not in "fiu":
        raise ValueError("Features must be a finite numeric (N,F) array")
    features = np.ascontiguousarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Features must remain finite in float32")
    if fingerprint(path) != before:
        raise ValueError("Input changed while being read")
    # Edge shape, integral dtype, range, reversal and duplicate handling are
    # checked by the shared clean-split implementation.
    return features, edges, before


def training_support(nodes, train_positive):
    """Exact two-hop+self support derived exclusively from training edges."""
    u, v = train_positive.T
    row, col = np.concatenate((u, v)), np.concatenate((v, u))
    adjacency = sp.csr_matrix((np.ones(len(row), dtype=np.float64), (row, col)),
                              shape=(nodes, nodes))
    step = adjacency + sp.eye(nodes, format="csr", dtype=np.float64)
    support = (step @ step).tocsr()
    support.sum_duplicates()
    support.sort_indices()
    support.eliminate_zeros()
    rows = np.repeat(np.arange(nodes, dtype=np.int64), np.diff(support.indptr))
    return np.stack((rows, support.indices.astype(np.int64)), axis=0)


def prepare(path, config, device):
    features, edges, input_record = load_arrays(path)
    split = make_clean_lp_split(len(features), edges, config["seed"],
        valid_fraction=config["valid_fraction"], test_fraction=config["test_fraction"])
    support = training_support(len(features), split.train_positive)
    feature_record = dict(shape=list(features.shape), dtype="float32",
        sha256=hashlib.sha256(features.tobytes()).hexdigest(),
        policy="caller_preprocessed_no_transform")
    support_record = dict(shape=list(support.shape),
        sha256=hashlib.sha256(support.astype("<i8").tobytes()).hexdigest(),
        policy="training_edges_only_two_hop_including_self")
    return (torch.from_numpy(features).to(device), torch.from_numpy(support).to(device),
            split, dict(input=input_record, features=feature_record,
                        support=support_record, split=split.provenance()))


def assert_finite(values, stage):
    for value in values:
        if value is not None and not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite tensor during {stage}")


def edge_scores(model, features, support, positive, negative):
    model.eval()
    with torch.no_grad():
        embedding = model.embed(features, support)
        logits = torch.cat((model.edge_logits(embedding, positive),
                            model.edge_logits(embedding, negative))).flatten()
    assert_finite([logits], "evaluation")
    return logits.detach().cpu()


def rank_metrics(logits, positive_count):
    values = logits.detach().cpu().numpy().reshape(-1)
    if not 0 < positive_count < len(values) or not np.isfinite(values).all():
        raise ValueError("Two finite edge classes are required")
    labels = np.concatenate((np.ones(positive_count), np.zeros(len(values) - positive_count)))
    return dict(roc_auc=float(roc_auc_score(labels, values)),
                average_precision=float(average_precision_score(labels, values)),
                positive_count=int(positive_count), negative_count=int(len(values) - positive_count))


def construct(config, feature_count, device):
    model, optimizer = build_model(config["arm"], config["seed"], feature_count, device,
        width=config["width"], heads=config["heads"], layers=config["layers"],
        dropout=config["dropout"])
    for group in optimizer.param_groups:
        group.update(lr=config["lr"], weight_decay=config["weight_decay"])
    return model, optimizer


def train(input_path, output, config=None, device="cpu"):
    config = validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Training output already exists; use a new directory")
    sources = source_fingerprints()
    features, support, split, provenance = prepare(input_path, config, device)
    output.mkdir(parents=True, exist_ok=False)
    model, optimizer = construct(config, features.shape[1], device)
    tensor = lambda array: torch.tensor(array.copy(), dtype=torch.long, device=device)
    positive = tensor(split.train_positive)
    valid_positive, valid_negative = tensor(split.valid_positive), tensor(split.valid_negative)
    history, best_state, best_auc, best_epoch, stale = [], None, -math.inf, None, 0
    for epoch in range(1, config["epochs"] + 1):
        sampled = split.training_negatives(epoch)
        negative = tensor(sampled)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embedding = model.embed(features, support)
        logits = torch.cat((model.edge_logits(embedding, positive),
                            model.edge_logits(embedding, negative))).flatten()
        labels = torch.cat((torch.ones(len(positive), device=device), torch.zeros(len(negative), device=device)))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        assert_finite([loss, logits], "training forward")
        loss.backward()
        assert_finite([p.grad for p in model.parameters()], "gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        assert_finite(model.parameters(), "optimizer update")
        validation = rank_metrics(edge_scores(model, features, support, valid_positive, valid_negative), len(valid_positive))
        if validation["roc_auc"] > best_auc:
            best_auc, best_epoch, stale = validation["roc_auc"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        history.append(dict(epoch=epoch, loss=float(loss.detach().cpu()), validation=validation,
            selected_epoch=best_epoch,
            training_negative_sha256=hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest()))
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state, strict=True)
    selected = rank_metrics(edge_scores(model, features, support, valid_positive, valid_negative), len(valid_positive))
    if selected != history[best_epoch - 1]["validation"]:
        raise ValueError("Selected checkpoint failed validation replay")
    if fingerprint(input_path) != provenance["input"] or source_fingerprints() != sources:
        raise ValueError("Input or source changed during fitting")
    record = dict(schema=SCHEMA, config=config, config_sha256=canonical_hash(config),
        provenance=provenance, sources=sources, history=history, selected_epoch=best_epoch,
        validation=selected, selection="first_strict_maximum_validation_ROC_AUC",
        test_scored=False, environment=runtime_metadata(device))
    with (output / "checkpoint.pt").open("xb") as stream:
        torch.save(dict(state_dict=best_state, record_sha256=canonical_hash(record)), stream)
    manifest = dict(record=record, checkpoint=fingerprint(output / "checkpoint.pt"))
    write_json(output / "manifest.json", manifest)
    return manifest


def evaluate(input_path, run, output, device="cpu", config=None):
    """Verify a frozen run and score its held-out pairs once through this runner.

    The exclusive run marker is an audit safeguard, not a security boundary.
    Failed scoring leaves the marker for inspection rather than silently retrying.
    """
    output, run = Path(output), Path(run)
    marker = run / "test_evaluation_started.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Evaluation output exists; refusing to overwrite test results")
    if marker.exists() or marker.is_symlink():
        raise FileExistsError("Test evaluation was already started for this run")
    protected = {run / name for name in ("manifest.json", "checkpoint.pt", "test_evaluation_started.json")}
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Evaluation output must not replace a frozen artifact or evaluation marker")
    manifest = read_json(run / "manifest.json")
    if set(manifest) != {"record", "checkpoint"}:
        raise ValueError("Invalid run manifest")
    record = manifest["record"]
    if record["schema"] != SCHEMA or record["test_scored"] is not False:
        raise ValueError("Expected an unscored portable training run")
    frozen = validate_config(record["config"])
    if frozen != record["config"] or canonical_hash(frozen) != record["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    if config is not None and validate_config(config) != frozen:
        raise ValueError("Supplied configuration differs from frozen training configuration")
    if record["sources"] != source_fingerprints():
        raise ValueError("Source files differ from the training run")
    if manifest["checkpoint"] != fingerprint(run / "checkpoint.pt"):
        raise ValueError("Checkpoint fingerprint mismatch")
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["record_sha256"] != canonical_hash(record):
        raise ValueError("Training record changed after checkpoint selection")
    features, support, split, provenance = prepare(input_path, frozen, device)
    if provenance != record["provenance"]:
        raise ValueError("Input, processed arrays, or split differ from training")
    model, _ = construct(frozen, features.shape[1], device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert_finite(model.parameters(), "checkpoint load")
    if fingerprint(input_path) != provenance["input"] or source_fingerprints() != record["sources"]:
        raise ValueError("Input or source changed before test evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, dict(record_sha256=checkpoint["record_sha256"],
        checkpoint=manifest["checkpoint"], input=provenance["input"], result_filename=output.name))
    tensor = lambda array: torch.tensor(array.copy(), dtype=torch.long, device=device)
    logits = edge_scores(model, features, support, tensor(split.test_positive), tensor(split.test_negative))
    result = dict(schema=SCHEMA, seed=frozen["seed"], arm=frozen["arm"],
        record_sha256=checkpoint["record_sha256"], checkpoint=manifest["checkpoint"],
        test=rank_metrics(logits, len(split.test_positive)),
        logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        device=str(device), environment=runtime_metadata(device))
    if fingerprint(input_path) != provenance["input"] or source_fingerprints() != record["sources"]:
        raise ValueError("Input or source changed during evaluation")
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--input", required=True, type=Path, help="Preprocessed features and edges NPZ")
        command.add_argument("--config", type=Path, help="JSON overrides; evaluate requires agreement with frozen config")
        command.add_argument("--device", default="cpu", help="Explicit torch device, e.g. cpu or cuda:0")
        command.add_argument("--output", required=name == "train", type=Path,
                             help="New run directory (train) or new result JSON (evaluate)")
        if name == "train":
            command.add_argument("--seed", type=int, help="Override the single run seed")
        if name == "evaluate":
            command.add_argument("--run", required=True, type=Path)
    args = parser.parse_args(argv)
    config = read_json(args.config) if args.config else None
    if args.command == "train":
        if args.seed is not None:
            config = ({} if config is None else config) | {"seed": args.seed}
        result = train(args.input, args.output, config, args.device)
        print(json.dumps(dict(output=str(args.output), validation=result["record"]["validation"], test_scored=False)))
    else:
        result = evaluate(args.input, args.run, args.output or args.run / "test_metrics.json", args.device, config)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Create a deterministic source archive from an explicit allowlist.

No datasets, weights, outputs, local environments, or Git history are exported.
Refuses an existing archive and refuses a stale existing manifest.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = (
    ".gitignore", "README.md", "NOTICE.md", "VALIDATION.md",
    "SOURCE_PROVENANCE.json", "pyproject.toml", "requirements.txt",
    "busemann_transformer.py", "paper_lp_model.py", "paper_lp_splits.py",
    "train_nc.py", "train_lp.py",
    "paper/checklist.tex",
)
PATTERNS = ("configs/*.json", "examples/*.py", "tests/*.py", "scripts/*.py")
PRIVATE = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    re.compile(r"/(?:gpfs|scratch)/[A-Za-z0-9_.-]+/"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
)


def selected_files():
    paths = {ROOT / name for name in TOP_LEVEL}
    for pattern in PATTERNS:
        paths.update(ROOT.glob(pattern))
    # Include an author-supplied license when one has actually been recovered.
    paths.update(p for p in (ROOT / "LICENSE", ROOT / "LICENSE.md") if p.exists())
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/nonregular release file: {path.name}")
        if any(p.is_symlink() for p in path.parents if p != ROOT.parent):
            raise ValueError("Symlinked release directory is not supported")
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def validate_sources(paths):
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in PRIVATE):
            raise ValueError(f"Review potential private material in {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            ast.parse(text, filename=str(path.relative_to(ROOT)))


def manifest_text(paths):
    return "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                   f"{path.relative_to(ROOT).as_posix()}\n" for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new archive path")
    paths = selected_files()
    validate_sources(paths)
    manifest = ROOT / "MANIFEST.sha256"
    expected = manifest_text(paths)
    if manifest.exists():
        if manifest.read_text() != expected:
            raise ValueError("Manifest is stale; review changes before regenerating it")
    else:
        with manifest.open("x") as stream:
            stream.write(expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*paths, manifest]:
            name = "busemannformer/" + path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Packaged {len(paths) + 1} files; SHA256={digest}")


if __name__ == "__main__":
    main()

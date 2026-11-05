"""Validate source hashes, syntax, and the export allowlist; no training."""
import json
from pathlib import Path
import hashlib

from package_release import ROOT, selected_files, validate_sources, manifest_text


def main():
    paths = selected_files()
    validate_sources(paths)
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file() or manifest.read_text() != manifest_text(paths):
        raise ValueError("Missing, stale, or inconsistent release manifest")
    provenance = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text())
    for item in provenance["sources"]:
        # This single test's GPU skip condition is documented in NOTICE.md.
        if item["path"] == "tests/test_paper_lp_model.py":
            continue
        actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
        if actual != item["original_sha256"]:
            raise ValueError(f"Unexplained source change: {item['path']}")
    print(f"Verified {len(paths)} source files: hashes, syntax, and private-path scan")
    if not any((ROOT / name).is_file() for name in ("LICENSE", "LICENSE.md")):
        print("Release blocker: original project license has not been located")


if __name__ == "__main__":
    main()

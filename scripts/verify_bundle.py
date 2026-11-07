#!/usr/bin/env python3
"""Check all downloaded artifacts and requirement locks without installing anything."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(root=ROOT):
    root = Path(root)
    wheels = json.loads((root / "provenance/wheels.json").read_text(encoding="utf-8"))
    for name, expected in wheels["lockfiles_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Requirement lock SHA256 mismatch: {name}")
    for item in wheels["wheels"]:
        path = root / "wheelhouse" / item["filename"]
        if path.stat().st_size != item["size"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Wheel SHA256 mismatch: {item['filename']}")
    upstream = json.loads((root / "provenance/upstream.json").read_text(encoding="utf-8"))
    if file_sha256(root / upstream["archive_path"]) != upstream["archive_sha256"]:
        raise ValueError("Author source archive SHA256 mismatch.")
    return {"package_count": wheels["package_count"], "wheel_bytes": wheels["total_wheel_bytes"],
            "source_commit": upstream["commit"], "source_archive_sha256": upstream["archive_sha256"],
            "wheel_hashes_verified": True, "lockfile_hashes_verified": True}


if __name__ == "__main__":
    print(json.dumps(verify_bundle(), indent=2))

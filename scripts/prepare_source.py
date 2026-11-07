#!/usr/bin/env python3
"""Verify the bundled author archive and apply two documented dependency edits.

No network calls. Existing modified source files are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def prepare_source(project_root=ROOT):
    project_root = Path(project_root).resolve()
    manifest = json.loads((project_root / "provenance/upstream.json").read_text(encoding="utf-8"))
    archive_path = project_root / manifest["archive_path"]
    if sha256(archive_path.read_bytes()) != manifest["archive_sha256"]:
        raise ValueError("Author archive SHA256 mismatch; re-upload the bundle.")
    files = manifest["upstream_files"]
    patches = {p["path"]: p for p in manifest["compatibility_patches"]}
    work = project_root / manifest["working_directory"]
    # First validate every archive member, then write only missing source files.
    with zipfile.ZipFile(archive_path) as archive:
        contents = {}
        for entry in archive.infolist():
            if not entry.filename.startswith(manifest["archive_prefix"]):
                raise ValueError("Unexpected archive root.")
            relative = entry.filename[len(manifest["archive_prefix"]):]
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise ValueError("Unsafe path in source archive.")
            if entry.is_dir() or not relative:
                continue
            if relative not in files or relative in contents:
                raise ValueError(f"Unexpected or repeated source file: {relative}")
            data = archive.read(entry)
            if len(data) != files[relative]["size"] or git_blob_sha1(data) != files[relative]["git_blob_sha1"]:
                raise ValueError(f"Git blob mismatch: {relative}")
            contents[relative] = data
        if contents.keys() != files.keys():
            raise ValueError("Missing files in author archive.")
        # Validate all existing files BEFORE changing anything.
        for relative, original in contents.items():
            target = work / relative
            if target.is_symlink():
                raise ValueError(f"Refusing source symlink: {relative}")
            if target.exists():
                actual = target.read_bytes()
                unchanged = actual == original
                patched = relative in patches and sha256(actual) == patches[relative]["patched_sha256"]
                if not (unchanged or patched):
                    raise ValueError(f"Source has local changes; preserving it: {relative}. Review before reinstalling.")
        for relative, original in contents.items():
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(original)
        for relative, patch in patches.items():
            target = work / relative
            original = target.read_bytes()
            if sha256(original) == patch["patched_sha256"]:
                continue
            if sha256(original) != patch["original_sha256"] or original.count(patch["old"].encode()) != 1:
                raise ValueError(f"Compatibility patch precondition failed: {relative}")
            patched = original.replace(patch["old"].encode(), patch["new"].encode())
            if sha256(patched) != patch["patched_sha256"]:
                raise ValueError(f"Compatibility patch hash mismatch: {relative}")
            target.write_bytes(patched)
    return {"commit": manifest["commit"], "archive_sha256": manifest["archive_sha256"],
            "verified_upstream_file_count": len(files), "working_directory": str(work),
            "compatibility_patched_files": list(patches), "algorithm_files_modified": False}


if __name__ == "__main__":
    print(json.dumps(prepare_source(), indent=2))

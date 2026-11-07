"""Offline, deterministic MRPC resource verification; no third-party imports."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": 3668, "validation": 408}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def row_identity(split, row):
    return {"sample_id": f"{split}:{row['idx']}", "idx": row["idx"],
            "row_sha256": hashlib.sha256(canonical(row).encode("utf-8")).hexdigest()}


def audit_rows(splits):
    if set(splits) != set(SPLITS):
        raise ValueError("Expected exactly train and validation; no new split or subsampling.")
    result = {"splits": {}}
    ids, pairs, unordered_pairs, sentences = {}, {}, {}, {}
    for split, expected in SPLITS.items():
        rows = splits[split]
        if len(rows) != expected:
            raise ValueError(f"{split}: expected {expected} rows, got {len(rows)}")
        for row in rows:
            if set(row) != {"sentence1", "sentence2", "label", "idx"}:
                raise ValueError(f"{split}: unexpected columns")
            if type(row["idx"]) is not int or row["idx"] < 0:
                raise ValueError(f"{split}: invalid native idx")
            if type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError(f"{split}: invalid binary label")
            if any(not isinstance(row[k], str) or not row[k].strip() for k in ("sentence1", "sentence2")):
                raise ValueError(f"{split}: empty or invalid sentence")
        ids[split] = {r["idx"] for r in rows}
        if len(ids[split]) != len(rows):
            raise ValueError(f"{split}: duplicate native idx")
        pair_list = [(r["sentence1"], r["sentence2"]) for r in rows]
        pairs[split] = set(pair_list)
        unordered_pairs[split] = {tuple(sorted(p)) for p in pair_list}
        sentences[split] = {s for pair in pair_list for s in pair}
        content = "".join(canonical(r) + "\n" for r in rows).encode("utf-8")
        result["splits"][split] = {
            "num_rows": len(rows), "label_counts": dict(sorted(Counter(str(r["label"]) for r in rows).items())),
            "unique_native_ids": len(ids[split]), "native_id_min": min(ids[split]), "native_id_max": max(ids[split]),
            "ordered_rows_sha256": hashlib.sha256(content).hexdigest(),
            "duplicate_ordered_pairs": len(rows) - len(pairs[split]),
            "duplicate_unordered_pairs": len(rows) - len(unordered_pairs[split])}
    overlap = ids["train"] & ids["validation"]
    if overlap:
        raise ValueError("Native IDs overlap between train and validation.")
    result["cross_split"] = {
        "native_id_overlap": len(overlap),
        "exact_ordered_pair_overlap": len(pairs["train"] & pairs["validation"]),
        "exact_unordered_pair_overlap": len(unordered_pairs["train"] & unordered_pairs["validation"]),
        "exact_sentence_overlap": len(sentences["train"] & sentences["validation"]),
        "interpretation": "Exact strings only; sentence overlap alone does not establish leakage. Original split is preserved."}
    return result


def verify_assets(root=ROOT):
    root = Path(root).resolve()
    manifest_path = root / "provenance/assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen = set()
    for item in manifest["files"]:
        rel = item["path"]
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or rel in seen:
            raise ValueError(f"Invalid or duplicate manifest path: {rel}")
        seen.add(rel)
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Asset SHA256 or size mismatch: {rel}")
    splits = {s: read_rows(root / "data/mrpc" / (s + ".jsonl")) for s in SPLITS}
    audit = audit_rows(splits)
    expected = json.loads((root / "data/mrpc/audit.json").read_text(encoding="utf-8"))
    if audit != expected:
        raise ValueError("MRPC audit differs from the fixed record.")
    identities = {s: [row_identity(s, r) for r in rows] for s, rows in splits.items()}
    if identities != json.loads((root / "data/mrpc/sample_ids.json").read_text(encoding="utf-8")):
        raise ValueError("MRPC sample identities/order differ from the fixed record.")
    return {"verified_file_count": len(seen), "manifest_sha256": sha256_file(manifest_path),
            "model_revision": manifest["model"]["revision"], "dataset_revision": manifest["dataset"]["revision"],
            "data_audit": audit}


if __name__ == "__main__":
    print(json.dumps(verify_assets(), ensure_ascii=False, indent=2))

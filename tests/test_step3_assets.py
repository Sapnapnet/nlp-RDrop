"""Checks for corrupt inputs, split mistakes and failure diagnostics before upload."""
import copy
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from asset_utils import audit_rows, read_rows, sha256_file, verify_assets


class Step3Assets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {s: read_rows(ROOT / f"data/mrpc/{s}.jsonl") for s in ("train", "validation")}

    def test_fixed_resources_and_audit(self):
        result = verify_assets()
        self.assertEqual(result["verified_file_count"], 17)
        audit = result["data_audit"]
        self.assertEqual(audit["splits"]["train"]["label_counts"], {"0": 1194, "1": 2474})
        self.assertEqual(audit["splits"]["validation"]["label_counts"], {"0": 129, "1": 279})
        self.assertEqual(audit["cross_split"]["native_id_overlap"], 0)

    def test_corrupted_file_fails_before_use(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "provenance").mkdir()
            item = root / "input.txt"
            item.write_text("before", encoding="utf-8")
            manifest = {"files": [{"path": "input.txt", "size": item.stat().st_size, "sha256": sha256_file(item)}]}
            (root / "provenance/assets.json").write_text(json.dumps(manifest), encoding="utf-8")
            item.write_text("edited", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_assets(root)

    def test_wrong_label_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows["train"][0]["label"] = -1
        with self.assertRaisesRegex(ValueError, "binary label"):
            audit_rows(rows)

    def test_cross_split_id_contamination_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows["validation"][0]["idx"] = rows["train"][0]["idx"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_rows(rows)

    def test_row_order_is_part_of_identity(self):
        original = audit_rows(self.rows)
        rows = copy.deepcopy(self.rows)
        rows["train"][0], rows["train"][1] = rows["train"][1], rows["train"][0]
        changed = audit_rows(rows)
        self.assertNotEqual(original["splits"]["train"]["ordered_rows_sha256"],
                            changed["splits"]["train"]["ordered_rows_sha256"])

    @unittest.skipIf(platform.system() == "Linux" and sys.version_info[:2] == (3, 10), "Unsupported-host test only")
    def test_wrong_host_produces_failure_feedback(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/check_step3.py"), "--output-root", scratch],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 1)
            outputs = list(Path(scratch).iterdir())
            self.assertEqual(len(outputs), 1)
            with zipfile.ZipFile(outputs[0] / "feedback_step3.zip") as archive:
                report = json.loads(archive.read("asset_report.json"))
                self.assertEqual(report["overall_status"], "FAIL")
                self.assertNotIn("command", report)
                self.assertIn("check.log", archive.namelist())


if __name__ == "__main__":
    unittest.main()

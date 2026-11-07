"""Offline provenance, non-overwrite and diagnostic checks (standard library only)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_source import prepare_source
from verify_bundle import verify_bundle


class Step2Artifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source_fixture(self):
        manifest = json.loads((ROOT / "provenance/upstream.json").read_text(encoding="utf-8"))
        (self.fixture / "provenance").mkdir()
        (self.fixture / "vendor").mkdir()
        shutil.copy2(ROOT / "provenance/upstream.json", self.fixture / "provenance/upstream.json")
        shutil.copy2(ROOT / manifest["archive_path"], self.fixture / manifest["archive_path"])
        return manifest

    def test_downloaded_bundle_matches_all_hashes(self):
        result = verify_bundle()
        self.assertEqual(result["package_count"], 44)
        self.assertTrue(result["wheel_hashes_verified"])

    def test_source_patch_is_repeatable_and_algorithm_bytes_unchanged(self):
        manifest = self.source_fixture()
        first = prepare_source(self.fixture)
        second = prepare_source(self.fixture)
        self.assertEqual(first, second)
        bert = "huggingface_transformer_src/src/transformers/models/bert/modeling_bert.py"
        with zipfile.ZipFile(self.fixture / manifest["archive_path"]) as z:
            self.assertEqual(z.read(manifest["archive_prefix"] + bert),
                             (self.fixture / manifest["working_directory"] / bert).read_bytes())
        self.assertEqual(first["verified_upstream_file_count"], 1213)

    def test_local_modifications_are_preserved_and_stop_extraction(self):
        manifest = self.source_fixture()
        target = self.fixture / manifest["working_directory"] / "README.md"
        target.parent.mkdir(parents=True)
        target.write_text("user changes to preserve\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "local changes"):
            prepare_source(self.fixture)
        self.assertEqual(target.read_text(), "user changes to preserve\n")
        self.assertFalse((target.parent / "huggingface_transformer_src").exists())

    def test_corrupt_archive_is_rejected_before_extraction(self):
        manifest = self.source_fixture()
        archive = self.fixture / manifest["archive_path"]
        with archive.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare_source(self.fixture)
        self.assertFalse((self.fixture / manifest["working_directory"]).exists())

    def test_modified_requirement_lock_is_rejected(self):
        (self.fixture / "provenance").mkdir()
        shutil.copy2(ROOT / "provenance/wheels.json", self.fixture / "provenance/wheels.json")
        (self.fixture / "requirements-step2.lock").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "lock SHA256"):
            verify_bundle(self.fixture)

    @unittest.skipIf(platform.system() == "Linux" and sys.version_info[:2] == (3,10),
                     "Unsupported-host failure test only; never install into a test machine")
    def test_unsupported_host_writes_failure_feedback_without_installing(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/setup_step2.py"),
                                 "--output-root", str(self.fixture / "reports")],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 1)
        reports = list((self.fixture / "reports").iterdir())
        self.assertEqual(len(reports), 1)
        report = json.loads((reports[0] / "install_result.json").read_text())
        self.assertEqual(report["overall_status"], "FAIL")
        self.assertEqual(report["commands"], [])
        with zipfile.ZipFile(reports[0] / "feedback_step2.zip") as z:
            self.assertIn("install_result.json", z.namelist())
            self.assertIn("install.log", z.namelist())


if __name__ == "__main__":
    unittest.main()

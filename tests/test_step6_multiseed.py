"""Stage-6 audits, statistics, failure/reuse control and tiny CPU training tests."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
os.environ.update({"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0"})
from baseline_multiseed_runtime import load_run_config
from baseline_summary import (EVIDENCE_FILES, binary_scores, load_json, summarize, verify_anchor,
                              verify_bundle, verify_run, write_json, write_summary)
from run_step6 import child_environment, package_feedback, run_remaining


class Stage6Tests(unittest.TestCase):
    def test_configs_only_change_seed_and_seed13_guard(self):
        base = load_json(ROOT / "configs/baseline_seed13.json")
        base.pop("validated_seed13_initial_state_sha256")
        for seed in (21, 42):
            base["seed"] = seed
            self.assertEqual(load_run_config(seed), base)
            self.assertEqual(child_environment(seed)["PYTHONHASHSEED"], str(seed))
            self.assertEqual(child_environment(seed)["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        with self.assertRaises(ValueError):
            load_run_config(13)

    def test_frozen_historical_files_and_new_bundle(self):
        self.assertEqual(verify_bundle(ROOT, "step5_bundle.json"), "00159813234ef27763bfda1f9894f755f23d26cd6c636574515957c8939a7566")
        self.assertEqual(len(verify_bundle(ROOT, "step6_bundle.json")), 64)

    def test_actual_accepted_seed13_independent_audit(self):
        result = verify_run(verify_anchor(), 13)
        self.assertEqual(result["accuracy"], 334 / 408)
        self.assertEqual(result["f1"], 524 / 598)
        self.assertEqual(result["optimizer_updates"], 2296)
        self.assertAlmostEqual(result["train_loop_wall_seconds"], 333.61797773092985)
        self.assertEqual(result["batch_plan_sha256"], "9c14bfd60b3eaf735f514a70e8e11792cb339adb87fd7ed6bd7579d5ba4cb379")
        self.assertFalse(result["checkpoint_files_verified_now"])

    def test_tampered_predictions_workload_and_status_rejected(self):
        source = verify_anchor()
        for kind in ("predictions", "workload", "status", "code_identity"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                for name in EVIDENCE_FILES:
                    shutil.copyfile(source / name, output / name)
                if kind in ("predictions", "workload"):
                    path = output / ("predictions_dev.jsonl" if kind == "predictions" else "training_updates.jsonl")
                    rows = [json.loads(s) for s in path.read_text().splitlines()]
                    if kind == "predictions":
                        rows[0]["prediction"] = 1 - rows[0]["prediction"]
                    else:
                        rows[0]["native_ids"][0] = -999
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                else:
                    path = output / "result.json"
                    result = load_json(path)
                    result["status" if kind == "status" else "code_manifest_sha256"] = "INVALID"
                    write_json(path, result)
                with self.assertRaises(RuntimeError):
                    verify_run(output, 13)

    def fixture_results(self):
        # Deliberately synthetic records, in memory only; never formal MRPC results.
        common = {"status": "PASS", "artifact_checks": "PASS", "protocol_id": "TEST_ONLY", "protocol_sha256": "p",
                  "input_manifest_sha256": "i", "training_core_sha256": "c", "condition": "baseline",
                  "optimizer_updates": 2296, "encoder_forward_calls": 4583, "encoder_example_passes": 36628,
                  "examples_seen": 36628, "checkpoint_selection": "last_update", "peak_allocated_bytes": 2**30,
                  "peak_reserved_bytes": 2*2**30}
        return [{**common, "seed": seed, "initial_state_sha256": str(seed), "accuracy": acc, "f1": f1,
                 "train_loop_wall_seconds": seconds, "mean_wall_seconds_per_update": seconds/2296}
                for seed, acc, f1, seconds in ((13,.7,.6,100), (21,.8,.75,200), (42,.9,.9,300))]

    def test_hand_calculated_mean_sample_sd_and_markdown_units(self):
        summary = summarize(list(reversed(self.fixture_results())))
        self.assertAlmostEqual(summary["mean_accuracy"], .8)
        self.assertAlmostEqual(summary["sample_std_accuracy"], .1)
        self.assertAlmostEqual(summary["mean_f1"], .75)
        self.assertAlmostEqual(summary["sample_std_f1"], .15)
        self.assertAlmostEqual(summary["mean_train_loop_wall_seconds"], 200)
        self.assertAlmostEqual(summary["sample_std_train_loop_wall_seconds"], 100)
        self.assertEqual(summary["std_ddof"], 1)
        with tempfile.TemporaryDirectory() as temp:
            write_summary(Path(temp), summary)
            text = (Path(temp) / "baseline_results.md").read_text(encoding="utf-8")
            self.assertIn("80.0000% ± 10.0000 个百分点", text)
            self.assertIn("75.0000% ± 15.0000 个百分点", text)
            self.assertEqual(load_json(Path(temp) / "baseline_results.json"), summary)

    def test_missing_duplicate_failed_incomparable_seeds_rejected(self):
        values = self.fixture_results()
        bad_sets = [values[:2], [values[0], values[0], values[2]]]
        for key, value in (("status", "FAIL"), ("protocol_sha256", "wrong"), ("examples_seen", 5), ("initial_state_sha256", "13")):
            bad = copy.deepcopy(values)
            bad[1][key] = value
            bad_sets.append(bad)
        for results in bad_sets:
            with self.assertRaises(RuntimeError):
                summarize(results)

    def test_binary_f1_positive_class_and_zero_division(self):
        scores = binary_scores([{"label": 1, "prediction": 1}, {"label": 0, "prediction": 1},
                                {"label": 1, "prediction": 0}, {"label": 0, "prediction": 0}])
        self.assertEqual(scores["accuracy"], .5)
        self.assertEqual(scores["f1"], .5)
        self.assertEqual(binary_scores([{"label": 0, "prediction": 0}])["f1"], 0)

    def test_sequential_launch_and_completed_runs_reused(self):
        calls = []
        def launch(seed, output):
            calls.append(seed)
            write_json(output / "result.json", {"seed": seed})
        def validate(output, seed):
            value = load_json(output / "result.json")
            self.assertEqual(value["seed"], seed)
            return value
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp)
            expected = [{"seed": 21}, {"seed": 42}]
            self.assertEqual(run_remaining(batch, launch, validate, lambda _: None), expected)
            self.assertEqual(calls, [21, 42])
            self.assertEqual(run_remaining(batch, launch, validate, lambda _: None), expected)
            self.assertEqual(calls, [21, 42])

    def test_failure_preserves_partial_run_and_stops_next_seed(self):
        calls = []
        def launch(seed, output):
            calls.append(seed)
            (output / "run.log").write_text("partial evidence")
            raise RuntimeError("injected training failure")
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run_remaining(batch, launch, lambda *_: None, lambda _: None)
            with self.assertRaisesRegex(RuntimeError, "partial artifacts"):
                run_remaining(batch, launch, lambda *_: None, lambda _: None)
            self.assertEqual(calls, [21])
            self.assertFalse((batch / "seed42").exists())
            self.assertEqual((batch / "seed21/run.log").read_text(), "partial evidence")

    def test_feedback_includes_all_seed_evidence_but_no_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp)
            write_json(batch / "step6_result.json", {"overall_status": "FAIL"})
            for seed in (13, 21, 42):
                (batch / f"seed{seed}/checkpoint-final").mkdir(parents=True)
                (batch / f"seed{seed}/checkpoint-final/pytorch_model.bin").write_bytes(b"not real weights")
                (batch / f"seed{seed}/run.log").write_text("fixture log")
            with zipfile.ZipFile(package_feedback(batch, batch / "seed13")) as archive:
                self.assertEqual(set(archive.namelist()), {"step6_result.json", "seed13/run.log", "seed21/run.log", "seed42/run.log"})

    @unittest.skipIf(sys.platform == "linux", "Unsupported-host check only")
    def test_wrong_host_fails_before_training_and_returns_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/run_step6.py"), "--output-dir", temp],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(load_json(Path(temp) / "launcher_failure.json")["overall_status"], "FAIL")
            self.assertTrue((Path(temp) / "feedback_step6.zip").is_file())
            self.assertFalse((Path(temp) / "seed21").exists())

    def test_two_seeds_tiny_cpu_pipeline_preserves_training_contract(self):
        import torch
        from transformers import BertConfig
        from src.baseline_model import BaselineBertForSequenceClassification
        from src.experiment_runner import run_experiment
        from src.training_core import seed_everything
        torch.set_num_threads(2)
        fingerprints, plans = [], []
        protocol = load_json(ROOT / "configs/training_protocol_v1.json")
        protocol.update(microbatch_size=4, gradient_accumulation_steps=2, eval_batch_size=4,
                        log_every_updates=2, train_precision="fp32_test_only")
        protocol["optimizer"]["learning_rate"] = .001
        def data(n):
            ids = torch.arange(n*8).reshape(n, 8) % 24 + 1
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": torch.zeros_like(ids), "labels": torch.arange(n) % 2}
        for seed in (21, 42):
            seed_everything(seed)
            model = BaselineBertForSequenceClassification(BertConfig(vocab_size=25, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=16, num_labels=2))
            config = load_run_config(seed)
            config.update(max_updates=7, warmup_updates=2, expected_microbatches=12, expected_training_example_visits=42,
                          expected_encoder_forward_calls=12, expected_encoder_example_passes=42)
            report = {"purpose": "local_tiny_cpu_test_not_a_formal_result"}
            with tempfile.TemporaryDirectory() as temp:
                run_experiment(model, None, {"train": data(17), "validation": data(11)},
                               {"train": list(range(17)), "validation": list(range(100,111))}, Path(temp),
                               protocol, config, torch.device("cpu"), report)
                self.assertEqual(report["overall_status"], "PASS")
                self.assertEqual(report["training"]["examples_seen"], 42)
                self.assertEqual(report["training"]["forward_counts"]["encoder_forward_calls"], 12)
                self.assertEqual(report["checkpoint_reload"]["max_abs_logit_difference"], 0)
                fingerprints.append(report["initial_state_sha256"])
                plans.append(report["batch_plan_sha256"])
        self.assertEqual(len(set(fingerprints)), 2)
        self.assertEqual(len(set(plans)), 2)


if __name__ == "__main__":
    unittest.main()

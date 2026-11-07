"""CPU integration tests for the full-run orchestration, using a tiny author BERT."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

os.environ.update({"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0"})
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import BertConfig
from src.baseline_model import BaselineBertForSequenceClassification
from src.training_core import make_update_plan, seed_everything
from src.experiment_runner import run_experiment, summarize_plan
from run_step5 import package_feedback, stream_child


def model():
    return BaselineBertForSequenceClassification(BertConfig(vocab_size=25, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=16, num_labels=2,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1))


def data(count):
    ids = torch.arange(count*8).reshape(count, 8) % 24 + 1
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": torch.zeros_like(ids),
            "labels": torch.arange(count) % 2}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(13)
        self.protocol = json.loads((ROOT / "configs/training_protocol_v1.json").read_text(encoding="utf-8"))
        self.protocol.update(microbatch_size=4, gradient_accumulation_steps=2, eval_batch_size=4, log_every_updates=2,
                             train_precision="fp32_test_only")
        self.protocol["optimizer"]["learning_rate"] = 0.001
        self.config = {"seed": 13, "max_updates": 7, "warmup_updates": 2, "expected_microbatches": 12,
                       "expected_training_example_visits": 42}
        self.tensors = {"train": data(17), "validation": data(11)}
        self.ids = {"train": list(range(17)), "validation": list(range(100,111))}

    def test_full_pipeline_tail_accounting_predictions_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            report = {"purpose": "local_cpu_test_not_mrpc_result"}
            progress = []
            run_experiment(model(), None, self.tensors, self.ids, output, self.protocol, self.config,
                           torch.device("cpu"), report, on_progress=progress.append)
            self.assertEqual(report["overall_status"], "PASS")
            self.assertEqual(report["training"]["optimizer_step_calls"], 7)
            self.assertEqual(report["training"]["microbatches"], 12)
            self.assertEqual(report["training"]["examples_seen"], 42)
            self.assertEqual(report["planned_workload"]["update_size_counts"], {"1":2, "8":5})
            self.assertEqual(report["forward_counts"]["eval"]["encoder_forward_calls"], 3)
            self.assertEqual(report["checkpoint_reload"]["max_abs_logit_difference"], 0)
            self.assertEqual([p["completed_updates"] for p in progress], [1,2,4,6,7])
            self.assertGreaterEqual(report["training"]["train_loop_wall_seconds"], report["training"]["sum_update_body_seconds"])
            predictions = [json.loads(line) for line in (output / "predictions_dev.jsonl").read_text().splitlines()]
            self.assertEqual([p["idx"] for p in predictions], self.ids["validation"])
            acc = sum(p["label"] == p["prediction"] for p in predictions) / 11
            self.assertEqual(acc, report["dev_metrics"]["accuracy"])
            events = [json.loads(line) for line in (output / "training_updates.jsonl").read_text().splitlines()]
            self.assertEqual([e["examples_seen"] for e in events], [8,8,1,8,8,1,8])
            self.assertEqual(sum(e["encoder_forward_calls"] for e in events), 12)
            checkpoint = json.loads((output / "checkpoint_manifest.json").read_text())
            self.assertFalse(checkpoint["optimizer_saved"])
            self.assertEqual(checkpoint["state_sha256"], report["final_state_sha256"])

    def test_exact_formal_workload_and_unchanged_optimizer_settings(self):
        p = json.loads((ROOT / "configs/training_protocol_v1.json").read_text(encoding="utf-8"))
        old = json.loads((ROOT / "configs/training_protocol.json").read_text(encoding="utf-8"))
        for key in ("optimizer", "scheduler", "microbatch_size", "gradient_accumulation_steps", "loss_contract", "determinism"):
            self.assertEqual(p[key], old[key])
        plan = make_update_plan(3668, 8, 2, 2296, 13)
        self.assertEqual(summarize_plan(plan), {"optimizer_updates":2296,"microbatches":4583,"example_visits":36628,
                                              "update_size_counts":{"4":9,"16":2287},"epochs_entered":10})

    def test_budget_mismatch_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            self.config["expected_microbatches"] = 999
            with self.assertRaisesRegex(RuntimeError, "Workload plan mismatch"):
                run_experiment(model(), None, self.tensors, self.ids, Path(temp), self.protocol, self.config,
                               torch.device("cpu"), {})
            self.assertFalse((Path(temp)/"training_updates.jsonl").exists())

    def test_failure_retains_partial_updates_without_success_marker(self):
        def interrupt(progress):
            raise RuntimeError("injected interruption")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                run_experiment(model(), None, self.tensors, self.ids, output, self.protocol, self.config,
                               torch.device("cpu"), {}, on_progress=interrupt)
            self.assertEqual(len((output/"training_updates.jsonl").read_text().splitlines()), 1)
            self.assertNotEqual(json.loads((output/"formal_report.json").read_text())["overall_status"], "PASS")
            self.assertFalse((output/"checkpoint-final").exists())

    def test_feedback_never_includes_large_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output/"run.log").write_text("log")
            (output/"result.json").write_text('{"status":"PASS"}')
            (output/"checkpoint-final").mkdir()
            (output/"checkpoint-final/pytorch_model.bin").write_bytes(b"weights")
            with zipfile.ZipFile(package_feedback(output)) as archive:
                self.assertEqual(set(archive.namelist()), {"run.log", "result.json"})

    def test_child_failure_is_streamed_and_propagated(self):
        lines = []
        with self.assertRaisesRegex(RuntimeError, "exited with 7"):
            stream_child([sys.executable, "-c", "print('child-diagnostic'); raise SystemExit(7)"], os.environ.copy(), lines.append, timeout=20)
        self.assertIn("child-diagnostic", lines)

    @unittest.skipIf(sys.platform == "linux", "Unsupported-host behavior only")
    def test_server_launcher_wrong_host_returns_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            process = subprocess.run([sys.executable, str(ROOT/"scripts/run_step5.py"), "--output-root", temp],
                                     capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(process.returncode, 1, process.stderr)
            with zipfile.ZipFile(next(Path(temp).glob("*/feedback_step5_seed13.zip"))) as archive:
                report = json.loads(archive.read("step5_result.json"))
                self.assertEqual(report["overall_status"], "FAIL")
                self.assertNotIn("command", report)


if __name__ == "__main__":
    unittest.main()

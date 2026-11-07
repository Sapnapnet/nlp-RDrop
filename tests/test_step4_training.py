"""Tiny author-BERT training tests; CPU-compatible and no network or pretrained assets."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile

os.environ.update({"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0"})
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import BertConfig
from sklearn.metrics import accuracy_score, f1_score
from src.baseline_model import BaselineBertForSequenceClassification
from src.training_core import (ForwardCounter, binary_metrics, evaluate, make_optimizer, make_scheduler,
                               make_update_plan, model_fingerprint, seed_everything, train_update)


def tiny(dropout=0.0):
    return BaselineBertForSequenceClassification(BertConfig(vocab_size=25, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=16, num_labels=2,
        hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout))


def data(n=6):
    ids = torch.arange(n * 8).reshape(n, 8) % 24 + 1
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": torch.zeros_like(ids),
            "labels": torch.arange(n) % 2}


OPT = {"learning_rate": 0.001, "betas": [0.9, 0.98], "eps": 1e-6, "weight_decay": 0.01}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        seed_everything(13)

    def test_baseline_is_single_forward_mean_ce_with_train_dropout(self):
        model = tiny(dropout=0.2)
        counter = ForwardCounter(model)
        counter.phase = "probe"
        model.train()
        x = data()
        first, second = model(**x), model(**x)
        self.assertFalse(torch.equal(first.logits, second.logits))
        self.assertTrue(torch.allclose(first.loss, torch.nn.functional.cross_entropy(first.logits, x["labels"])))
        self.assertEqual(counter.counts["probe"]["encoder_forward_calls"], 2)
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(**x).logits, model(**x).logits))
        counter.close()

    def test_weighted_accumulation_matches_full_batch_even_for_unequal_microbatches(self):
        for groups in ([[0, 1, 2, 3], [4, 5]], [[0, 1, 2, 3, 4, 5]]):
            # FP64 isolates weighting correctness from Adam's amplification of FP32
            # roundoff when full and split batches use different matrix shapes.
            model = tiny().double()
            reference = copy.deepcopy(model)
            x = data()
            # A single full-batch reference uses an independent training path.
            refopt = make_optimizer(reference, OPT)
            reference.train()
            reference(**x).loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            refopt.step()
            opt = make_optimizer(model, OPT)
            schedule = make_scheduler(opt, 4, 0, 1e-7)
            counter = ForwardCounter(model)
            event = train_update(model, x, {"microbatches": groups, "epoch": 0}, opt, schedule,
                                 torch.cuda.amp.GradScaler(enabled=False), torch.device("cpu"), 1.0, False, 1, counter)
            self.assertEqual(event["examples_seen"], 6)
            self.assertEqual(event["optimizer_step_calls"], 1)
            for p, q in zip(model.parameters(), reference.parameters()):
                self.assertTrue(torch.allclose(p, q, atol=1e-12, rtol=1e-10), (p - q).abs().max().item())
            counter.close()

    def test_twenty_real_optimizer_steps_then_trained_checkpoint_reload(self):
        model = tiny(dropout=0.1)
        before = model_fingerprint(model)
        x = data(32)
        opt = make_optimizer(model, OPT)
        schedule = make_scheduler(opt, 20, 2, 1e-7)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        counter = ForwardCounter(model)
        events = [train_update(model, x, group, opt, schedule, scaler, torch.device("cpu"), 1.0, False, 1, counter)
                  for group in make_update_plan(32, 8, 2, 20, 13)]
        self.assertEqual(sum(e["optimizer_step_calls"] for e in events), 20)
        self.assertEqual(sum(e["examples_seen"] for e in events), 320)
        self.assertEqual(counter.counts["train"]["encoder_forward_calls"], 40)
        self.assertNotEqual(model_fingerprint(model), before)
        metrics, logits = evaluate(model, x, 7, torch.device("cpu"), counter)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored = BaselineBertForSequenceClassification.from_pretrained(folder, local_files_only=True)
            self.assertEqual(model_fingerprint(restored), model_fingerprint(model))
            counter2 = ForwardCounter(restored)
            metrics2, logits2 = evaluate(restored, x, 7, torch.device("cpu"), counter2)
            self.assertTrue(torch.equal(logits, logits2))
            self.assertEqual(metrics["accuracy"], metrics2["accuracy"])
            self.assertEqual(metrics["f1"], metrics2["f1"])
            counter2.close()
        counter.close()

    def test_epoch_tail_and_sampling_do_not_consume_dropout_rng(self):
        random.seed(13)
        before = random.getstate()
        torch_before = torch.get_rng_state().clone()
        plan = make_update_plan(3668, 8, 2, 2296, 13)
        self.assertEqual(random.getstate(), before)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(plan, make_update_plan(3668, 8, 2, 2296, 13))
        first_epoch = [idx for u in plan if u["epoch"] == 0 for b in u["microbatches"] for idx in b]
        self.assertEqual(sorted(first_epoch), list(range(3668)))
        self.assertEqual(sum(len(b) for b in plan[229]["microbatches"]), 4)
        self.assertEqual(sum(len(b) for u in plan for b in u["microbatches"]), 36628)
        self.assertEqual(sum(len(u["microbatches"]) for u in plan), 4583)

    def test_scheduler_matches_warmup_and_polynomial_boundary(self):
        model = tiny()
        opt = make_optimizer(model, OPT)
        schedule = make_scheduler(opt, 20, 2, 1e-7)
        used = []
        for _ in range(20):
            used.append(opt.param_groups[0]["lr"])
            opt.step()
            schedule.step()
        self.assertEqual(used[0], 0)
        self.assertAlmostEqual(used[1], OPT["learning_rate"] / 2)
        self.assertAlmostEqual(used[2], OPT["learning_rate"])
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-7)

    def test_checkpoint_roundtrip_keeps_single_forward_adapter_and_metrics(self):
        model = tiny()
        x = data(7)
        counter = ForwardCounter(model)
        before_metrics, before_logits = evaluate(model, x, 3, torch.device("cpu"), counter)
        self.assertEqual(counter.counts["eval"]["encoder_forward_calls"], 3)
        before = model_fingerprint(model)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored, info = BaselineBertForSequenceClassification.from_pretrained(folder, local_files_only=True, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])
            self.assertEqual(info["unexpected_keys"], [])
            self.assertEqual(model_fingerprint(restored), before)
            c2 = ForwardCounter(restored)
            after_metrics, after_logits = evaluate(restored, x, 3, torch.device("cpu"), c2)
            self.assertTrue(torch.equal(before_logits, after_logits))
            self.assertEqual(before_metrics["f1"], after_metrics["f1"])
            c2.close()
        counter.close()

    def test_metrics_match_sklearn_including_zero_positive_predictions(self):
        for labels, predictions in [([0, 1, 1, 0], [0, 1, 0, 0]), ([0, 0], [0, 0]), ([0, 1], [0, 0]), ([1, 1], [1, 1])]:
            result = binary_metrics(labels, predictions)
            self.assertEqual(result["accuracy"], accuracy_score(labels, predictions))
            self.assertEqual(result["f1"], f1_score(labels, predictions, zero_division=0))

    def test_same_seed_initializes_identical_model(self):
        first = model_fingerprint(tiny())
        seed_everything(13)
        self.assertEqual(model_fingerprint(tiny()), first)

    @unittest.skipIf(sys.platform == "linux", "Never run server launcher inside this CPU failure test")
    def test_wrong_host_packages_feedback_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/check_step4.py"), "--output-root", folder],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            with zipfile.ZipFile(next(Path(folder).glob("*/feedback_step4.zip"))) as archive:
                report = json.loads(archive.read("step4_result.json"))
                self.assertEqual(report["overall_status"], "FAIL")
                self.assertNotIn("command", report)


if __name__ == "__main__":
    unittest.main()

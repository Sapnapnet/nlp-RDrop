import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import BertConfig

from src.baseline_model import BaselineBertForSequenceClassification
from src.rdrop_model import RDropBertForSequenceClassification


torch.set_num_threads(1)


def make_config(alpha=None):
    config = BertConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_dropout_prob=0.25,
        attention_probs_dropout_prob=0.25,
        num_labels=2,
    )

    if alpha is not None:
        config.rdrop_alpha = float(alpha)

    return config


def make_batch():
    return {
        "input_ids": torch.arange(1, 49).reshape(4, 12),
        "attention_mask": torch.ones(4, 12, dtype=torch.long),
        "token_type_ids": torch.zeros(4, 12, dtype=torch.long),
        "labels": torch.tensor([0, 1, 0, 1]),
    }


def manual_rdrop_loss(logits1, logits2, labels, alpha):
    ce1 = F.cross_entropy(logits1.float(), labels)
    ce2 = F.cross_entropy(logits2.float(), labels)

    log_p1 = F.log_softmax(logits1.float(), dim=-1)
    log_p2 = F.log_softmax(logits2.float(), dim=-1)

    p1 = F.softmax(logits1.float(), dim=-1)
    p2 = F.softmax(logits2.float(), dim=-1)

    kl12 = F.kl_div(
        log_p1, p2, reduction="none"
    ).sum(dim=-1).mean()

    kl21 = F.kl_div(
        log_p2, p1, reduction="none"
    ).sum(dim=-1).mean()

    ce_loss = 0.5 * (ce1 + ce2)
    kl_loss = 0.5 * (kl12 + kl21)

    return ce_loss + alpha * kl_loss


class TestRDropModel(unittest.TestCase):

    def test_training_formula_and_gradients(self):
        """Check loss formula, two forwards and both-logit gradients."""

        batch = make_batch()

        for alpha in (0.0, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                torch.manual_seed(2026)

                model = RDropBertForSequenceClassification(
                    make_config(alpha)
                )
                model.train()

                forward_calls = {"count": 0}
                captured_logits = []

                def count_forward(module, inputs, output):
                    forward_calls["count"] += 1

                def capture_logits(module, inputs, output):
                    output.retain_grad()
                    captured_logits.append(output)

                h1 = model.bert.register_forward_hook(count_forward)
                h2 = model.classifier.register_forward_hook(
                    capture_logits
                )

                try:
                    outputs = model(**batch)

                    self.assertEqual(forward_calls["count"], 2)
                    self.assertEqual(len(captured_logits), 2)

                    logits1, logits2 = captured_logits

                    self.assertTrue(torch.isfinite(outputs.loss).item())

                    # Independent Dropout realizations.
                    self.assertFalse(
                        torch.equal(
                            logits1.detach(),
                            logits2.detach()
                        )
                    )

                    # Compare implementation with independent formula.
                    expected = manual_rdrop_loss(
                        logits1,
                        logits2,
                        batch["labels"],
                        alpha,
                    )

                    torch.testing.assert_close(
                        outputs.loss.detach(),
                        expected.detach(),
                        rtol=1e-5,
                        atol=1e-6,
                    )

                    # Check gradients produced by the actual model loss.
                    outputs.loss.backward()

                    self.assertIsNotNone(logits1.grad)
                    self.assertIsNotNone(logits2.grad)

                    self.assertTrue(torch.isfinite(logits1.grad).all())
                    self.assertTrue(torch.isfinite(logits2.grad).all())

                    self.assertGreater(
                        logits1.grad.abs().sum().item(), 0
                    )
                    self.assertGreater(
                        logits2.grad.abs().sum().item(), 0
                    )

                    # Independently verify gradients against the formula.
                    clone1 = logits1.detach().clone().requires_grad_(True)
                    clone2 = logits2.detach().clone().requires_grad_(True)

                    expected_clone = manual_rdrop_loss(
                        clone1,
                        clone2,
                        batch["labels"],
                        alpha,
                    )

                    grad1, grad2 = torch.autograd.grad(
                        expected_clone,
                        (clone1, clone2),
                    )

                    torch.testing.assert_close(
                        logits1.grad, grad1, rtol=1e-5, atol=1e-6
                    )
                    torch.testing.assert_close(
                        logits2.grad, grad2, rtol=1e-5, atol=1e-6
                    )

                    # Check classifier and encoder receive gradients.
                    self.assertIsNotNone(model.classifier.weight.grad)
                    self.assertIsNotNone(
                        model.bert.embeddings.word_embeddings.weight.grad
                    )

                    self.assertGreater(
                        model.classifier.weight.grad.abs().sum().item(), 0
                    )

                    self.assertTrue(
                        torch.isfinite(
                            model.bert.embeddings.word_embeddings.weight.grad
                        ).all()
                    )

                finally:
                    h1.remove()
                    h2.remove()

                print(
                    f"PASS: training alpha={alpha}, "
                    f"two forwards, loss and gradients"
                )

    def test_evaluation_single_forward(self):
        """Evaluation must use exactly one encoder forward."""

        model = RDropBertForSequenceClassification(
            make_config(0.5)
        )
        model.eval()

        count = {"value": 0}

        def hook(module, inputs, output):
            count["value"] += 1

        handle = model.bert.register_forward_hook(hook)

        try:
            with torch.no_grad():
                outputs = model(**make_batch())

            self.assertEqual(count["value"], 1)
            self.assertEqual(outputs.logits.shape, (4, 2))
            self.assertTrue(torch.isfinite(outputs.loss).item())
            self.assertTrue(torch.isfinite(outputs.logits).all())

        finally:
            handle.remove()

        print("PASS: evaluation uses exactly one forward")

    def test_same_seed_initialization_as_baseline(self):
        """Baseline and R-Drop constructors must initialize identically."""

        torch.manual_seed(1234)
        baseline = BaselineBertForSequenceClassification(
            make_config()
        )

        torch.manual_seed(1234)
        rdrop = RDropBertForSequenceClassification(
            make_config(0.5)
        )

        baseline_state = baseline.state_dict()
        rdrop_state = rdrop.state_dict()

        self.assertEqual(
            list(baseline_state.keys()),
            list(rdrop_state.keys()),
        )

        for key in baseline_state:
            torch.testing.assert_close(
                baseline_state[key],
                rdrop_state[key],
                rtol=0,
                atol=0,
            )

        print("PASS: identical initialization under the same seed")

    def test_checkpoint_save_and_reload(self):
        """Alpha and model weights must survive save/load."""

        torch.manual_seed(2026)

        model = RDropBertForSequenceClassification(
            make_config(0.5)
        )
        model.eval()

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)

            restored, info = (
                RDropBertForSequenceClassification.from_pretrained(
                    directory,
                    local_files_only=True,
                    output_loading_info=True,
                )
            )

            self.assertEqual(info["missing_keys"], [])
            self.assertEqual(info["unexpected_keys"], [])
            self.assertEqual(info["error_msgs"], [])

            self.assertAlmostEqual(restored.rdrop_alpha, 0.5)
            self.assertAlmostEqual(
                restored.config.rdrop_alpha, 0.5
            )

            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    value,
                    restored.state_dict()[key],
                    rtol=0,
                    atol=0,
                )

            restored.eval()

            with torch.no_grad():
                original_output = model(**make_batch())
                restored_output = restored(**make_batch())

            torch.testing.assert_close(
                original_output.logits,
                restored_output.logits,
                rtol=0,
                atol=0,
            )

        print("PASS: checkpoint and alpha save/reload")

    def test_invalid_alpha(self):
        """Negative and nonfinite alpha values must be rejected."""

        for alpha in (-0.5, float("nan"), float("inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    RDropBertForSequenceClassification(
                        make_config(alpha)
                    )

        print("PASS: invalid alpha values are rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)

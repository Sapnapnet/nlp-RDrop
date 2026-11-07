"""R-Drop adapter for the pinned author's BERT sequence classifier.

Training:
    Two independent stochastic forwards.
    0.5 * (CE1 + CE2) + alpha * 0.5 * (KL12 + KL21)

Evaluation:
    One deterministic forward, ordinary cross entropy.

The pinned vendor source is never modified.
"""

import math

import torch
import torch.nn.functional as F

from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class RDropBertForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)

        alpha = float(getattr(config, "rdrop_alpha", 1.0))

        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("rdrop_alpha must be finite and nonnegative.")

        if self.num_labels < 2:
            raise ValueError("This R-Drop adapter requires classification.")

        self.rdrop_alpha = alpha

        # Persist alpha in Hugging Face config.json.
        self.config.rdrop_alpha = alpha

    def _forward_once(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return outputs, logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # Evaluation: exactly one encoder forward.
        if not self.training:

            outputs, logits = self._forward_once(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            loss = None

            if labels is not None:
                loss = F.cross_entropy(
                    logits.float().reshape(-1, self.num_labels),
                    labels.reshape(-1),
                )

            if not return_dict:
                result = (logits,) + outputs[2:]
                return ((loss,) + result) if loss is not None else result

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Training: two independent stochastic encoder forwards.
        outputs1, logits1 = self._forward_once(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        _, logits2 = self._forward_once(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None

        if labels is not None:

            # CE: mean over examples.
            ce1 = F.cross_entropy(
                logits1.float().reshape(-1, self.num_labels),
                labels.reshape(-1),
            )

            ce2 = F.cross_entropy(
                logits2.float().reshape(-1, self.num_labels),
                labels.reshape(-1),
            )

            # KL calculations in FP32 for numerical stability.
            log_p1 = F.log_softmax(logits1.float(), dim=-1)
            log_p2 = F.log_softmax(logits2.float(), dim=-1)

            p1 = F.softmax(logits1.float(), dim=-1)
            p2 = F.softmax(logits2.float(), dim=-1)

            # KL(P2 || P1), batchmean.
            kl12 = F.kl_div(
                log_p1,
                p2,
                reduction="none",
            ).sum(dim=-1).mean()

            # KL(P1 || P2), batchmean.
            kl21 = F.kl_div(
                log_p2,
                p1,
                reduction="none",
            ).sum(dim=-1).mean()

            # Frozen experiment protocol.
            ce_loss = 0.5 * (ce1 + ce2)
            kl_loss = 0.5 * (kl12 + kl21)

            loss = ce_loss + self.rdrop_alpha * kl_loss

        if not return_dict:
            result = (logits1,) + outputs1[2:]
            return ((loss,) + result) if loss is not None else result

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )

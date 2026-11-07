"""Select the author's existing single-forward CE path without editing its source."""
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class BaselineBertForSequenceClassification(BertForSequenceClassification):
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        loss, logits, outputs = self.bert_cross_entropy(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask,
            inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        if not return_dict:
            result = (logits,) + outputs[2:]
            return ((loss,) + result) if loss is not None else result
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=outputs.hidden_states, attentions=outputs.attentions)

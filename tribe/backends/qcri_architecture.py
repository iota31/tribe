"""QCRI Fine-Grained Propaganda Detection model architecture.

Source: QCRI/PropagandaTechniquesAnalysis-en-BERT
Paper: Da San Martino et al., "Fine-Grained Analysis of Propaganda in News Article", EMNLP 2019

This module defines the custom BertForTokenAndSequenceJointClassification model
used by the QCRI classifier. The model requires 'bert-base-cased' tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput

# QCRI technique detection model identifier
QCRI_MODEL = "QCRI/PropagandaTechniquesAnalysis-en-BERT"

# The 20 token-level labels (index 0=PAD, index 1=O, indices 2-19 = techniques)
QCRI_TOKEN_TAGS = (
    "<PAD>",  # 0
    "O",  # 1 - not propaganda
    "Name_Calling,Labeling",  # 2
    "Repetition",  # 3
    "Slogans",  # 4
    "Appeal_to_fear-prejudice",  # 5
    "Doubt",  # 6
    "Exaggeration,Minimisation",  # 7
    "Flag-Waving",  # 8
    "Loaded_Language",  # 9
    "Reductio_ad_hitlerum",  # 10
    "Bandwagon",  # 11
    "Causal_Oversimplification",  # 12
    "Obfuscation,Intentional_Vagueness,Confusion",  # 13
    "Appeal_to_Authority",  # 14
    "Black-and-White_Fallacy",  # 15
    "Thought-terminating_Cliches",  # 16
    "Red_Herring",  # 17
    "Straw_Men",  # 18
    "Whataboutism",  # 19
)

# Sequence-level tags (binary: propaganda or not)
QCRI_SEQUENCE_TAGS = ("Non-prop", "Prop")

# Mapping from raw QCRI token tags to the 18 standard QCRI technique labels
QCRI_LABEL_MAP: dict[str, str] = {
    "Name_Calling,Labeling": "Name Calling/Labeling",
    "Repetition": "Repetition",
    "Slogans": "Slogans",
    "Appeal_to_fear-prejudice": "Appeal to Fear/Prejudice",
    "Doubt": "Doubt",
    "Exaggeration,Minimisation": "Exaggeration/Minimisation",
    "Flag-Waving": "Flag-Waving",
    "Loaded_Language": "Loaded Language",
    "Reductio_ad_hitlerum": "Appeal to Authority",
    "Bandwagon": "Bandwagon",
    "Causal_Oversimplification": "Causal Oversimplification",
    "Obfuscation,Intentional_Vagueness,Confusion": "Intentional Vagueness",
    "Appeal_to_Authority": "Appeal to Authority",
    "Black-and-White_Fallacy": "Black-and-White Fallacy",
    "Thought-terminating_Cliches": "Thought-Terminating Cliche",
    "Red_Herring": "Whataboutism/Red Herring",
    "Straw_Men": "Doubt",
    "Whataboutism": "Whataboutism/Red Herring",
}


@dataclass
class TokenAndSequenceJointClassifierOutput(ModelOutput):
    """Output from the joint token and sequence classifier."""

    loss: Optional[Tensor] = None
    token_logits: Optional[Tensor] = None
    sequence_logits: Optional[Tensor] = None
    hidden_states: Optional[tuple[Tensor, ...]] = None
    attentions: Optional[tuple[Tensor, ...]] = None


class BertForTokenAndSequenceJointClassification(BertPreTrainedModel):
    """BERT model for joint token-level and sequence-level propaganda detection.

    The model has two classification heads:
    - Token-level: classifies each token as one of 20 labels (O, PAD, or 18 techniques)
    - Sequence-level: classifies the entire sequence as propaganda or not
    """

    model_prefix = "bert"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_token_labels = 20
        self.num_sequence_labels = 2
        self.token_tags = QCRI_TOKEN_TAGS
        self.sequence_tags = QCRI_SEQUENCE_TAGS
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([
            nn.Linear(config.hidden_size, self.num_token_labels),
            nn.Linear(config.hidden_size, self.num_sequence_labels),
        ])
        self.masking_gate = nn.Linear(2, 1)
        self.init_weights()
        # Additional unused head from original model (kept for weight compatibility)
        self.merge_classifier_1 = nn.Linear(
            self.num_token_labels + self.num_sequence_labels, self.num_token_labels
        )

    def forward(  # type: ignore[override]
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
        return_dict=True,
    ) -> TokenAndSequenceJointClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs[0]
        pooler_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        token_logits = self.classifier[0](sequence_output)

        pooler_output = self.dropout(pooler_output)
        sequence_logits = self.classifier[1](pooler_output)

        gate = torch.sigmoid(self.masking_gate(sequence_logits))
        gates = gate.unsqueeze(1).repeat(1, token_logits.size(1), token_logits.size(2))
        weighted_token_logits = torch.mul(gates, token_logits)

        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            criterion = CrossEntropyLoss(ignore_index=0)
            weighted_flat = weighted_token_logits.view(-1, weighted_token_logits.shape[-1])
            token_loss = criterion(weighted_flat, labels.view(-1))
            loss = token_loss

        if not return_dict:
            output = (weighted_token_logits, sequence_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenAndSequenceJointClassifierOutput(
            loss=loss,
            token_logits=weighted_token_logits,
            sequence_logits=sequence_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

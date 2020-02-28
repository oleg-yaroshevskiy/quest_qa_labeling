import torch
import transformers
from bert import BertPreTrainedModel, BertModel
from torch import nn
from transformers import RobertaModel


class BertForQuestRegression(BertPreTrainedModel):
    def __init__(self, config, head_dropout=None):
        super(BertForQuestRegression, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        if head_dropout is None:
            head_dropout = config.hidden_dropout_prob

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        pooled_output = torch.mean(sequence_output, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def load(self, checkpoint, strict=True, **cfg_args):
        self.config.__dict__.update(cfg_args)
        self.__init__(self.config)

        state_dict = torch.load(checkpoint)
        return self.load_state_dict(state_dict, strict=strict)


class RobertaForQuestRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestRegression, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.roberta(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        pooled_output = torch.mean(sequence_output, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def load(self, checkpoint, strict=True, **cfg_args):
        self.config.__dict__.update(cfg_args)
        self.__init__(self.config)

        state_dict = torch.load(checkpoint)
        return self.load_state_dict(state_dict, strict=strict)


class CustomBert(transformers.BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(CustomBert, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        hidden_layers = outputs[2]

        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        return logits


def get_optimizer(model, learning_rate, backbone_prefix="bert"):
    params = list(model.named_parameters())

    def is_backbone(name):
        return backbone_prefix in name

    optimizer_grouped_parameters = [
        {"params": [p for n, p in params if is_backbone(n)], "lr": learning_rate},
        {
            "params": [p for n, p in params if not is_backbone(n)],
            "lr": learning_rate * 500,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=learning_rate, weight_decay=0
    )

    return optimizer

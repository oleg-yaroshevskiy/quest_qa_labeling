from fairseq import utils
from fairseq.data import encoders
from fairseq.models.bart import BARTModel

import torch
import torch.nn.functional as F
from torch import nn


class CustomBART(nn.Module):
    def __init__(self, model_name, num_labels, num_hidden_layers=12, hidden_size=1024):
        super(CustomBART, self).__init__()
        self.num_labels = num_labels
        self.bart = BARTModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None):

        hidden_layers = self.bart.extract_features(input_ids, return_all_hiddens=True)

        cls_outputs = torch.stack(
            [self.dropout(layer[:, -1, :]) for layer in hidden_layers], dim=2
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

        outputs = logits
        # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)


def get_model(args):
    model = CustomBART(args.bert_model, num_labels=args.num_classes)
    model.cuda()
    model = nn.DataParallel(model)

    return model

import logging
from transformers.modeling_bert import BertPreTrainedModel
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from fairseq import utils
from fairseq.data import encoders
from fairseq.models.bart import BARTModel

logging.getLogger("transformers").setLevel(logging.ERROR)
import torch
import torch.nn.functional as F
from torch import nn


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


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


class BARTTokenizer:
    @classmethod
    def hub_models(cls):
        return {
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
        }

    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.task = task
        self.bpe = encoders.build_bpe(args)
        self.max_positions = 1024

    def encode(
        self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).
        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).
        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`
        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::
            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(" ")) > self.max_positions - 2:
            tokens = " ".join(tokens.split(" ")[: self.max_positions - 2])
        bpe_sentence = "<s> " + tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return cls(x["args"], x["task"])


def get_model_optimizer(args):
    model = CustomBART(args.bert_model, num_labels=args.num_classes)
    model.cuda()
    model = nn.DataParallel(model)
    params = list(model.named_parameters())

    def is_backbone(n):
        return "bart" in n

    optimizer_grouped_parameters = [
        {"params": [p for n, p in params if is_backbone(n)], "lr": args.lr},
        {"params": [p for n, p in params if not is_backbone(n)], "lr": args.lr * 500},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.lr, weight_decay=0
    )

    return model, optimizer

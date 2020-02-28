import os
from copy import deepcopy
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from callbacks import CSVParamLogger
from poutyne.framework import Model, ModelCheckpoint
from poutyne.framework.callbacks import Callback
from poutyne.framework.metrics import EpochMetric
from pytorch_transformers import BertConfig, BertForPreTraining
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import BertTokenizer
from utils import torch_to_numpy

from data import QuestDataset

TARGETS = [
    "question_score",
    "question_views",
    "question_favs",
    "answer_score",
    "is_answer_accepted",
]

# change paths here!
PATH_TO_DATA = Path("input")
PATH_TO_CKPT_CONFIG = Path("step1_lm_finetuning/data/")

# for a toy example, change to 600000 for real LM training
LEN_TO_SAMPLE = 50
SEED = 17
BATCH_SIZE = 1
NUM_WORKERS = 8
# change to some 20, need to track loss as well
N_EPOCHS = 3
LRATE = 1e-5
BATCHES_PER_STEP = 32

# the trained LM will be saved here
checkpoint_dir = PATH_TO_DATA / "stackx-base-cased"
stackx_data = pd.read_csv(
    PATH_TO_DATA / "qa_stackexchange_cleaned.csv", nrows=LEN_TO_SAMPLE
)

stackx_data["question_title"] = stackx_data["question_title"].astype(str)
stackx_data["question_body"] = stackx_data["question_body"].astype(str)
stackx_data["answer"] = stackx_data["answer"].astype(str)


class QuestMLMDataset(QuestDataset):
    def __init__(
        self,
        data_df,
        tokenizer,
        max_seg_length=512,
        answer_ratio=0.5,
        use_title=True,
        use_body=True,
        use_answer=True,
        title_col="question_title",
        body_col="question_body",
        answer_col="answer",
        mlm_probability=0.15,
        non_masked_idx=-1,
        padding_idx=0,
        sop_prob=0.5,
        target_cols=TARGETS,
    ):
        super(QuestMLMDataset, self).__init__(
            data_df=data_df,
            tokenizer=tokenizer,
            max_seg_length=max_seg_length,
            target_cols=target_cols,
            answer_ratio=answer_ratio,
            use_title=use_title,
            use_body=use_body,
            use_answer=use_answer,
            title_col=title_col,
            body_col=body_col,
            answer_col=answer_col,
        )
        self.mlm_probability = mlm_probability
        self.sop_prob = sop_prob
        self.non_masked_idx = non_masked_idx
        self.padding_idx = padding_idx
        self.cls_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.mask_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    def _mask_tokens(self, inputs, masked_random_replace_prob=0.2):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        masked_indices = torch.bernoulli(
            torch.full(labels.shape, self.mlm_probability)
        ).bool()
        for special_token in [self.cls_token_idx, self.sep_token_idx, self.padding_idx]:
            masked_indices &= inputs != special_token
        labels[
            ~masked_indices
        ] = self.non_masked_idx  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, 1 - masked_random_replace_prob)
            ).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.mask_token_idx
        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __getitem__(self, index):
        title, body, answer = self._get_text(index)
        numeric_targets = torch.FloatTensor(self.targets[index].astype(np.float32))
        sop_label = 0
        if np.random.uniform(0, 1) < self.sop_prob:
            sop_label = 1
            permutation = list(np.random.permutation(range(3)))
            if permutation == [0, 1, 2]:
                permutation = [0, 2, 1]
            title, body, answer = [[title, body, answer][i] for i in permutation]
        input_ids, attention_mask, token_type_ids = self._process(title, body, answer)
        input_ids, attention_mask, token_type_ids = map(
            torch.LongTensor, [input_ids, attention_mask, token_type_ids]
        )
        input_ids, labels = self._mask_tokens(torch.LongTensor(input_ids))
        return (
            (input_ids, token_type_ids, attention_mask),
            (labels, sop_label, numeric_targets),
        )


# Normalize aux targets


encoded = []
trange = tqdm(stackx_data["host"].unique())
for host in trange:
    host_mask = stackx_data["host"] == host
    trange.set_description(str(host))
    host_labels = deepcopy(stackx_data[host_mask][TARGETS])
    for col in ["question_score", "question_views", "question_favs", "answer_score"]:
        host_labels[col] = rankdata(stackx_data[host_mask][col]) / host_mask.sum()
    encoded.append(host_labels)

encoded = pd.concat(encoded, sort=False).reindex(stackx_data.index)
stackx_data[encoded.columns] = encoded

train_df, test_df = train_test_split(stackx_data, test_size=0.1, random_state=SEED)

tokenizer = BertTokenizer(
    str(PATH_TO_CKPT_CONFIG / "vocab.txt"), do_basic_tokenize=True, do_lower_case=False
)

train_dataset = QuestMLMDataset(train_df, tokenizer, target_cols=TARGETS)
val_dataset = QuestMLMDataset(test_df, tokenizer, target_cols=TARGETS)


class BertPretrain(BertForPreTraining):
    def __init__(self, config, num_labels):
        super(BertPretrain, self).__init__(config,)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        next_sentence_label=None,
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

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        mean_pooled_output = torch.mean(sequence_output, dim=1)
        mean_pooled_output = self.dropout(mean_pooled_output)
        logits = self.classifier(mean_pooled_output)

        outputs = (prediction_scores, seq_relationship_score, logits)
        return outputs


config = BertConfig(str(PATH_TO_CKPT_CONFIG / "config.json"))
model = BertPretrain(config, len(TARGETS))

# Prepare extended bert embedding
orig_bert = BertForPreTraining.from_pretrained("bert-base-cased")
orig_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

state_dict = orig_bert.state_dict()
del state_dict["cls.predictions.decoder.weight"], state_dict["cls.predictions.bias"]

orig_embedding = state_dict["bert.embeddings.word_embeddings.weight"]

extra_tokens = list(tokenizer.vocab.keys())[len(orig_tokenizer.vocab) :]
new_tokens_as_orig_indices = [[i] for i in range(len(orig_tokenizer.vocab))] + [
    orig_tokenizer.encode(t, add_special_tokens=False) for t in extra_tokens
]

new_embedding = torch.zeros(len(new_tokens_as_orig_indices), orig_embedding.shape[-1])
new_embedding.normal_(mean=0.0, std=0.02)

for row, indices in enumerate(new_tokens_as_orig_indices):
    if len(indices) > 0:
        new_embedding[row] = orig_embedding[indices].mean(0)

state_dict["bert.embeddings.word_embeddings.weight"] = new_embedding

# Load original pretrained weight with extended embedding layer
model.load_state_dict(state_dict, strict=False)
model.tie_weights()

device = "cuda"

steps_per_epoch = LEN_TO_SAMPLE // BATCH_SIZE

sampler = RandomSampler(
    train_dataset,
    num_samples=steps_per_epoch * BATCH_SIZE if steps_per_epoch is not None else None,
    replacement=True if steps_per_epoch is not None else False,
)

if NUM_WORKERS is None:
    NUM_WORKERS = cpu_count()

train_loader = DataLoader(
    train_dataset, sampler=sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)


def spearman_metric(y_true, y_pred, return_scores=False, colnames=None):
    corr = [
        spearmanr(pred_col, target_col).correlation
        for pred_col, target_col in zip(y_pred.T, y_true.T)
    ]
    if colnames is not None:
        return pd.Series(corr, index=colnames)
    if return_scores:
        return corr
    else:
        return np.nanmean(corr)


class Spearman(EpochMetric):
    class SpearmanCallback(Callback):
        def __init__(self):
            self.metric_values = dict()

        def on_epoch_end(self, epoch, logs):
            logs.update(self.metric_values)

    def __init__(self, colnames=None):
        super(Spearman, self).__init__()
        self.__name__ = "spearman"
        self.preds, self.targets = [], []

        self.colnames = colnames
        self.callback = self.SpearmanCallback()

    def forward(self, logits, targets):
        y_pred, y_true = logits[2], targets[2]
        self.preds.append(torch_to_numpy(y_pred))
        self.targets.append(torch_to_numpy(y_true))

    def get_metric(self):
        corr = spearman_metric(
            np.vstack(self.targets), np.vstack(self.preds), return_scores=True
        )

        if self.colnames is not None:
            self.callback.metric_values = dict(zip(self.colnames, corr))

        self.preds, self.targets = [], []
        return np.mean(corr)


class MaskLMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, logits, targets):
        n_samples = np.prod(targets.shape)
        loss = super(MaskLMCrossEntropyLoss, self).forward(
            logits.view(n_samples, -1), targets.view(n_samples)
        )
        return loss


class SOPCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, logits, targets):
        loss = super(SOPCrossEntropyLoss, self).forward(
            logits.view(-1, 2), targets.view(-1)
        )
        return loss


class PretrainingLoss(torch.nn.Module):
    def __init__(self, targets_alpha=1.0):
        super(PretrainingLoss, self).__init__()
        self.mlm_loss = MaskLMCrossEntropyLoss(ignore_index=-1)
        self.sop_loss = SOPCrossEntropyLoss()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.targets_alpha = targets_alpha

    def forward(self, logits, targets):
        return (
            self.mlm_loss(logits[0], targets[0])
            + self.sop_loss(logits[1], targets[1])
            + self.targets_alpha * self.bce(logits[2], targets[2])
        )


class MaskLMPerplexity(MaskLMCrossEntropyLoss):
    __name__ = "mlm_perplexity"

    def forward(self, logits, targets):
        logits, targets = logits[0], targets[0]
        loss = super(MaskLMPerplexity, self).forward(logits, targets)
        perplexity = 2 ** loss
        return float(perplexity)


def sop_accuracy(logits, targets):
    logits, targets = logits[1], targets[1]
    pred = torch.argmax(logits.view(-1, 2), dim=-1)
    targets = targets.view(-1)
    return float(torch.mean((pred == targets).float()))


spearman = Spearman(TARGETS)

trainer = Model(
    model,
    AdamW(model.parameters(), lr=LRATE),
    loss_function=PretrainingLoss(),
    batch_metrics=[MaskLMPerplexity(ignore_index=-1), sop_accuracy],
    epoch_metrics=[spearman],
)

if not checkpoint_dir.exists():
    os.makedirs(str(checkpoint_dir))

checkpoint_name = "stackx_base_with_aux_ep_{epoch:03}_val_perplexity_{val_mlm_perplexity:.2f}_val_spearman_{val_spearman:.2f}.pth"

callbacks = [
    spearman.callback,
    ModelCheckpoint(str(checkpoint_dir / checkpoint_name)),
    CSVParamLogger(str(checkpoint_dir / "training_log.csv"), extra_metrics=TARGETS),
]
trainer.to(device)

history = trainer.fit_generator(
    train_loader,
    val_loader,
    epochs=N_EPOCHS,
    batches_per_step=BATCHES_PER_STEP,
    callbacks=callbacks,
)

torch.cuda.empty_cache()

# prepare the latest checkpoint for further finetuning
latest_ckpt = sorted(list(checkpoint_dir.glob("stackx_base_with_au*")))[-1]
state_dict = torch.load(latest_ckpt)
# we'll be adapting classification for 30 targets, so delete classifier weights
del state_dict["classifier.bias"]
del state_dict["classifier.weight"]
torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
# also need config and vocabulary to reuse the model
os.system(f"cp {str(PATH_TO_CKPT_CONFIG / 'vocab.txt')} {str(checkpoint_dir)}")
os.system(f"cp {str(PATH_TO_CKPT_CONFIG / 'config.json')} {str(checkpoint_dir)}")

from math import floor, ceil

import torch
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import GroupKFold
import numpy as np
from tqdm import tqdm


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens)  # + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments  # + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids  # + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _trim_input(
    args,
    tokenizer,
    title,
    question,
    answer,
    max_sequence_length=290,
    t_max_len=30,
    q_max_len=128,
    a_max_len=128,
):
    # SICK THIS IS ALL SEEMS TO BE SICK

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError(
                "New sequence length should be %d, but is %d"
                % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4))
            )
        q_len_head = round(q_new_len / 2)
        q_len_tail = -1 * (q_new_len - q_len_head)
        a_len_head = round(a_new_len / 2)
        a_len_tail = -1 * (a_new_len - a_len_head)  ## Head+Tail method .
        t = t[:t_new_len]
        if args.head_tail:
            q = q[:q_len_head] + q[q_len_tail:]
            a = a[:a_len_head] + a[a_len_tail:]
        else:
            q = q[:q_new_len]
            a = a[:a_new_len]  ## No Head+Tail ,usual processing

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def _get_stoken_output(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    return stoken


def compute_input_arays(
    args,
    df,
    columns,
    tokenizer,
    max_sequence_length,
    t_max_len=30,
    q_max_len=128,
    a_max_len=128,
):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(
        df[columns].iterrows(), desc="Preparing dataset", total=len(df), ncols=80,
    ):
        t, q, a = (
            instance.question_title,
            instance.question_body,
            instance.answer,
        )
        t, q, a = _trim_input(
            args,
            tokenizer,
            t,
            q,
            a,
            max_sequence_length,
            t_max_len,
            q_max_len,
            a_max_len,
        )
        ids, masks, segments = _convert_to_bert_inputs(
            t, q, a, tokenizer, max_sequence_length
        )

        input_ids.append(np.array(ids, dtype=np.int64))
        input_masks.append(np.array(masks, dtype=np.int64))
        input_segments.append(np.array(segments, dtype=np.int64))

    return (input_ids, input_masks, input_segments)


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


class BucketingSampler:
    def __init__(self, lengths, batch_size, maxlen=500):

        self.lengths = lengths
        self.batch_size = batch_size
        self.maxlen = 500

        self.batches = self._make_batches(lengths, batch_size, maxlen)

    def _make_batches(self, lengths, batch_size, maxlen):

        max_total_length = maxlen * batch_size
        ids = np.argsort(lengths)

        current_maxlen = 0
        batch = []
        batches = []

        for id in ids:
            current_len = len(batch) * current_maxlen
            size = lengths[id]
            current_maxlen = max(size, current_maxlen)
            new_len = current_maxlen * (len(batch) + 1)
            if new_len < max_total_length:
                batch.append(id)
            else:
                batches.append(batch)
                current_maxlen = size
                batch = [id]

        if batch:
            batches.append(batch)

        assert (sum(len(batch) for batch in batches)) == len(lengths)

        return batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def make_collate_fn(
    padding_values={"input_ids": 0, "input_masks": 0, "input_segments": 0}
):
    def _collate_fn(batch):

        for name, padding_value in padding_values.items():

            lengths = [len(sample[name]) for sample in batch]
            max_length = max(lengths)

            for n, size in enumerate(lengths):
                p = max_length - size
                if p:
                    pad_width = [(0, p)] + [(0, 0)] * (batch[n][name].ndim - 1)
                    if padding_value == "edge":
                        batch[n][name] = np.pad(batch[n][name], pad_width, mode="edge")
                    else:
                        batch[n][name] = np.pad(
                            batch[n][name],
                            pad_width,
                            mode="constant",
                            constant_values=padding_value,
                        )

        return default_collate(batch)

    return _collate_fn


class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None):
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths

    @classmethod
    def from_frame(cls, args, df, tokenizer, test=False):
        """ here I put major preprocessing. why not lol
        """
        inputs = compute_input_arays(
            args,
            df,
            args.input_columns,
            tokenizer,
            max_sequence_length=args.max_sequence_length,
            t_max_len=args.max_title_length,
            q_max_len=args.max_question_length,
            a_max_len=args.max_answer_length,
        )

        outputs = None
        if not test:
            outputs = compute_output_arrays(df, args.target_columns)
            outputs = torch.tensor(outputs, dtype=torch.float32)

        # lengths = np.argmax(inputs[0] == 0, axis=1)
        # lengths[lengths == 0] = inputs[0].shape[1]
        lengths = [len(x) for x in inputs[0]]

        return cls(inputs=inputs, lengths=lengths, labels=outputs)

    def __len__(self):
        return len(self.inputs[0])

    def __getitem__(self, idx):
        input_ids = self.inputs[0][idx]
        input_masks = self.inputs[1][idx]
        input_segments = self.inputs[2][idx]
        lengths = self.lengths[idx]

        sample = dict(
            idx=idx,
            input_ids=input_ids,
            input_masks=input_masks,
            input_segments=input_segments,
            lengths=lengths,
        )

        if self.labels is not None:
            labels = self.labels[idx]
            sample["labels"] = labels

        return sample


def cross_validation_split(args, train_df, tokenizer, ignore_train=False):
    kf = GroupKFold(n_splits=args.folds)
    y_train = train_df[args.target_columns].values

    for fold, (train_index, val_index) in enumerate(
        kf.split(train_df.values, groups=train_df.question_title)
    ):

        if args.use_folds is not None and fold not in args.use_folds:
            continue

        if not ignore_train:
            train_subdf = train_df.iloc[train_index]
            train_set = QuestDataset.from_frame(args, train_subdf, tokenizer)
        else:
            train_set = None

        valid_set = QuestDataset.from_frame(args, train_df.iloc[val_index], tokenizer)

        yield (
            fold,
            train_set,
            valid_set,
            train_df.iloc[train_index],
            train_df.iloc[val_index],
        )


def get_pseudo_set(args, pseudo_df, tokenizer):
    return QuestDataset.from_frame(args, pseudo_df, tokenizer)


def get_test_set(args, test_df, tokenizer):
    return QuestDataset.from_frame(args, test_df, tokenizer, True)

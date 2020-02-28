from math import floor, ceil

import torch
from sklearn.model_selection import GroupKFold, KFold
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import html

BOS = 0
PAD = 1
EOS = 2


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    input_ids = tokens + [PAD] * (max_seq_length - len(tokens))
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

    t = tokenizer.encode(title)[1:-1].tolist()
    q = tokenizer.encode(question)[1:-1].tolist()
    a = tokenizer.encode(answer)[1:-1].tolist()

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

    stoken = [BOS] + title + [EOS] + question + [EOS] + answer + [EOS]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)

    return input_ids


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

    for col in ["question_title", "question_body", "answer"]:
        df[col] = df[col].apply(html.unescape)

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
        ids = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)

        input_ids.append(ids)

    return torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long()


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


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

        lengths = np.argmax(inputs == 0, axis=1)
        lengths[lengths == 0] = inputs.shape[1]

        return cls(inputs=inputs, lengths=lengths, labels=outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        lengths = self.lengths[idx]

        if self.labels is not None:
            labels = self.labels[idx]
            return input_ids, labels, lengths

        return input_ids, lengths


def cross_validation_split(
    args, train_df, tokenizer, ignore_train=False, pseudo_df=None, split_pseudo=False,
):
    kf = GroupKFold(n_splits=args.folds)
    y_train = train_df[args.target_columns].values

    leak_free_pseudo = isinstance(pseudo_df, list)

    if pseudo_df is not None:

        if leak_free_pseudo:
            n_pseudo = len(pseudo_df[0])
        else:
            n_pseudo = len(pseudo_df)

        if split_pseudo:
            pseudo_kfold = KFold(args.folds)
            pseudo_ids = iter(pseudo_kfold.split(np.arange(n_pseudo)))
        else:
            pseudo_ids = iter(
                [(np.arange(len(np.pseudo)), np.arange(len(n_pseudo)))] * args.folds
            )

    for fold, (train_index, val_index) in enumerate(
        kf.split(train_df.values, groups=train_df.question_title)
    ):
        if not ignore_train:
            train_subdf = train_df.iloc[train_index]

            if pseudo_df is not None:
                pseudo_train, pseudo_valid = next(pseudo_ids)

                pseudo_subdf = pseudo_df if not leak_free_pseudo else pseudo_df[fold]

                pseudo_subdf = pseudo_subdf.iloc[pseudo_valid]
                train_subdf = pd.concat([train_subdf, pseudo_subdf], sort=True)

            train_set = QuestDataset.from_frame(args, train_subdf, tokenizer)
        else:
            train_set = None
        valid_set = QuestDataset.from_frame(args, train_df.iloc[val_index], tokenizer)

        yield (
            train_set,
            valid_set,
            train_df.iloc[train_index],
            train_df.iloc[val_index],
        )


def get_pseudo_set(args, pseudo_df, tokenizer):
    return QuestDataset.from_frame(args, pseudo_df, tokenizer)


def get_test_set(args, test_df, tokenizer):
    return QuestDataset.from_frame(args, test_df, tokenizer, True)

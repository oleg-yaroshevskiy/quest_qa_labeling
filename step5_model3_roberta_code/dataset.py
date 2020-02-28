from math import floor, ceil
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold


def _get_masks(tokens, tokenizer, max_seq_length):
    """Mask for padding"""
    # if len(tokens) > max_seq_length:
    #     raise IndexError("Token length more than max seq length!")
    tokens = tokens[:max_seq_length]
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, tokenizer, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""

    # if len(tokens) > max_seq_length:
    #     raise IndexError("Token length more than max seq length!")
    tokens = tokens[:max_seq_length]

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == tokenizer.sep_token:
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = token_ids[:max_seq_length]
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
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


def _convert_to_bert_inputs(
    title, question, answer, tokenizer, max_sequence_length, model_type,
):
    """Converts tokenized input to ids, masks and segments for BERT"""
    if model_type == "roberta":
        stoken = (
            [tokenizer.cls_token]
            + title
            + [tokenizer.sep_token] * 2
            + question
            + [tokenizer.sep_token] * 2
            + answer
            + [tokenizer.sep_token] * 2
        )

    else:
        stoken = (
            [tokenizer.cls_token]
            + title
            + [tokenizer.sep_token]
            + question
            + [tokenizer.sep_token]
            + answer
            + [tokenizer.sep_token]
        )

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, tokenizer, max_sequence_length)
    input_segments = _get_segments(stoken, tokenizer, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(
    args,
    df,
    columns,
    tokenizer,
    max_sequence_length,
    t_max_len=30,
    q_max_len=128,
    a_max_len=128,
    verbose=True,
):
    input_ids, input_masks, input_segments = [], [], []

    iterator = df[columns].iterrows()
    if verbose:
        iterator = tqdm(iterator, desc="Preparing dataset", total=len(df), ncols=80,)

    for _, instance in iterator:
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
            t, q, a, tokenizer, max_sequence_length, args.model_type,
        )

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return (
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    )


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


class QuestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        df,
        tokenizer,
        test=False,
        title_transform=None,
        body_transform=None,
        answer_transform=None,
    ):
        self.data = df
        self.tokenizer = tokenizer
        self.is_test = test
        self.params = args

        self.title_transform = title_transform
        self.body_transform = body_transform
        self.answer_transform = answer_transform

    @classmethod
    def from_frame(
        cls,
        args,
        df,
        tokenizer,
        test=False,
        title_transform=None,
        body_transform=None,
        answer_transform=None,
    ):
        return cls(
            args,
            df,
            tokenizer,
            test=test,
            title_transform=title_transform,
            body_transform=body_transform,
            answer_transform=answer_transform,
        )

    def __getitem__(self, idx):
        instance = self.data[idx : idx + 1].copy()

        for col, transform in zip(
            ["question_title", "question_body", "answer"],
            [self.title_transform, self.body_transform, self.answer_transform],
        ):
            if transform is not None:
                instance[col] = transform[col]

        input_ids, input_masks, input_segments = compute_input_arrays(
            self.params,
            instance,
            self.params.input_columns,
            self.tokenizer,
            max_sequence_length=self.params.max_sequence_length,
            t_max_len=self.params.max_title_length,
            q_max_len=self.params.max_question_length,
            a_max_len=self.params.max_answer_length,
            verbose=False,
        )
        length = torch.sum(input_masks != 0)
        input_ids, input_masks, input_segments = (
            torch.squeeze(input_ids, dim=0),
            torch.squeeze(input_masks, dim=0),
            torch.squeeze(input_segments, dim=0),
        )

        if not self.is_test:
            labels = compute_output_arrays(instance, self.params.target_columns)
            labels = torch.tensor(labels, dtype=torch.float32)
            labels = torch.squeeze(labels, dim=0)
            return input_ids, input_masks, input_segments, labels, length
        else:
            return input_ids, input_masks, input_segments, length

    def __len__(self):
        return len(self.data)


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

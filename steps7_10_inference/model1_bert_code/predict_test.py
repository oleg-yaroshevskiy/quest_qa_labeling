import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from typing import List, Text


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


QUESTION_TARGETS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
]
ANSWER_TARGETS = [
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]
ALL_TARGETS = QUESTION_TARGETS + ANSWER_TARGETS


class QuestDataset(Dataset):
    def __init__(
        self,
        data_df,
        tokenizer,
        max_seg_length=256,
        target_cols="all_targets",
        answer_ratio=0.5,
        title_ratio=0.5,
        use_title=True,
        use_body=True,
        use_answer=True,
        title_col="question_title",
        body_col="question_body",
        answer_col="answer",
        title_transform=None,
        body_transform=None,
        answer_transform=None,
    ):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seg_length = max_seg_length
        self.target_cols = (
            QUESTION_TARGETS + ANSWER_TARGETS
            if target_cols is "all_targets"
            else target_cols
        )
        self.answer_ratio = answer_ratio
        self.title_ratio = title_ratio

        if target_cols is not None:
            if target_cols is "all_targets":
                target_cols = ALL_TARGETS
            self.targets = data_df[target_cols].values

        self.question_title = data_df[title_col].values if use_title else None
        self.question_body = data_df[body_col].values if use_body else None
        self.answer = data_df[answer_col].values if use_answer else None

        self.title_transform = title_transform
        self.body_transform = body_transform
        self.answer_transform = answer_transform

    def _encode_segments(self, *text_segments: List[Text]) -> List[List[int]]:
        # if self.transform is not None:
        #     text_segments = [self.transform(txt) for txt in text_segments]
        return [
            self.tokenizer.encode(
                txt, max_length=self.max_seg_length, add_special_tokens=False
            )
            if txt is not None
            else []
            for txt in text_segments
        ]

    def _process(self, title=None, body=None, answer=None):
        input_ids, attention_mask, token_type_ids = self._prepare_features(
            title, body, answer
        )

        input_ids = self._pad_and_truncate(
            input_ids, pad_value=self.tokenizer.pad_token_id
        )
        token_type_ids = self._pad_and_truncate(
            token_type_ids, pad_value=token_type_ids[-1]
        )
        attention_mask = self._pad_and_truncate(attention_mask, pad_value=0)
        return input_ids, attention_mask, token_type_ids

    def _pad_and_truncate(self, features, pad_value=0):
        features = list(features[: self.max_seg_length])
        features = features + [pad_value,] * (self.max_seg_length - len(features))
        features = np.array(features)
        return features

    @staticmethod
    def _balance_segments(
        first_segment_length, second_segment_length, second_ratio, max_length
    ):
        first_segment_length = min(
            first_segment_length,
            (1 - second_ratio) * max_length
            + max(second_ratio * max_length - second_segment_length, 0),
        )

        second_segment_length = min(
            second_segment_length,
            second_ratio * max_length
            + max((1 - second_ratio) * max_length - first_segment_length, 0),
        )

        return int(first_segment_length), int(second_segment_length)

    def _prepare_features(self, title, body, answer):
        title_input_ids, body_input_ids, answer_input_ids = self._encode_segments(
            title, body, answer
        )

        title_length = len(title_input_ids)
        body_length = len(body_input_ids)
        answer_length = len(answer_input_ids)

        question_length, answer_length = self._balance_segments(
            title_length + body_length,
            answer_length,
            self.answer_ratio,
            self.max_seg_length,
        )

        title_length, body_length = self._balance_segments(
            title_length, body_length, self.title_ratio, question_length
        )

        # TODO: generalize this
        question_input_ids = body_input_ids[:body_length]
        if title_length > 0:
            question_input_ids = (
                title_input_ids[:title_length]
                + [self.tokenizer.sep_token_id]
                + question_input_ids
            )
        answer_input_ids = answer_input_ids[:answer_length]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            question_input_ids, answer_input_ids if answer_length > 0 else None
        )
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
            question_input_ids, answer_input_ids
        )
        attention_mask = [1.0] * len(input_ids)

        return input_ids, attention_mask, token_type_ids

    def _get_text(self, index):
        title = self.question_title[index] if self.question_title is not None else None
        body = self.question_body[index] if self.question_body is not None else None
        answer = self.answer[index] if self.answer is not None else None

        def apply_transform(txt, transform):
            if transform is not None:
                return transform(txt, idx=index)
            else:
                return txt

        title, body, answer = [
            apply_transform(txt, transform)
            for txt, transform in zip(
                [title, body, answer],
                [self.title_transform, self.body_transform, self.answer_transform],
            )
        ]

        return title, body, answer

    def __getitem__(self, index):
        title, body, answer = self._get_text(index)

        input_ids, attention_mask, token_type_ids = self._process(title, body, answer)
        targets = self.targets[index]

        input_ids, attention_mask, token_type_ids = map(
            torch.LongTensor, [input_ids, attention_mask, token_type_ids]
        )
        targets = torch.FloatTensor(targets)

        return (input_ids, attention_mask, token_type_ids), targets

    def __len__(self):
        if self.answer is not None:
            return len(self.answer)
        elif self.question_title is not None:
            return len(self.question_title)
        else:
            return len(self.question_body)


class TestQuestDataset(QuestDataset):
    def __init__(
        self,
        data_df,
        tokenizer,
        max_seg_length=512,
        answer_ratio=0.5,
        title_ratio=0.5,
        use_title=True,
        use_body=True,
        use_answer=True,
        title_col="question_title",
        body_col="question_body",
        answer_col="answer",
    ):
        super(TestQuestDataset, self).__init__(
            data_df=data_df,
            tokenizer=tokenizer,
            max_seg_length=max_seg_length,
            target_cols=None,
            answer_ratio=answer_ratio,
            title_ratio=title_ratio,
            use_title=use_title,
            use_body=use_body,
            use_answer=use_answer,
            title_col=title_col,
            body_col=body_col,
            answer_col=answer_col,
        )

    def __getitem__(self, index):
        title, body, answer = self._get_text(index)

        input_ids, attention_mask, token_type_ids = self._process(title, body, answer)
        input_ids, attention_mask, token_type_ids = map(
            torch.LongTensor, [input_ids, attention_mask, token_type_ids]
        )
        return (input_ids, attention_mask, token_type_ids)


def predict(model, test_loader, columns, device="cuda"):
    model.eval()
    model.to(device)
    preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pred = torch_to_numpy(model(*torch_to(batch, device)))
            preds.append(pred)

    preds = np.vstack(preds)

    preds = torch.sigmoid(torch.from_numpy(preds)).numpy()
    preds = np.clip(preds, 0, 1 - 1e-8)
    preds = pd.DataFrame(preds, columns=columns)
    return preds


def torch_to_numpy(obj, copy=False):
    if copy:
        func = lambda t: t.cpu().detach().numpy().copy()
    else:
        func = lambda t: t.cpu().detach().numpy()
    return torch_apply(obj, func)


def torch_to(obj, *args, **kargs):
    return torch_apply(obj, lambda t: t.to(*args, **kargs))


def torch_apply(obj, func):
    fn = lambda t: func(t) if torch.is_tensor(t) else t
    return _apply(obj, fn)


def _apply(obj, func):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k: _apply(el, func) for k, el in obj.items()}
    return func(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, default="/kaggle/input/google-quest-challenge/"
    )
    parser.add_argument(
        "--model_dir", type=Path, default="/kaggle/input/stackx-80-aux-ep-3"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sub_file", type=str, default="submission.csv")
    args = parser.parse_args()

    def get_model(targets=ALL_TARGETS):
        config = BertConfig.from_json_file(
            args.model_dir / "stackx-base-cased-config.json"
        )
        config.__dict__["num_labels"] = len(targets)

        model = BertForQuestRegression(config)
        return model

    def predict_test_checkpoints(checkpoints, test_loader, targets, device="cuda"):
        model = get_model(targets)
        pred = []

        for path in checkpoints:
            model.load(path, map_location="cpu")
            pred.append(predict(model, test_loader, targets, device=device))

        pred = np.mean([p[targets].values for p in pred], axis=0)
        pred = np.clip(pred, 0, 1 - 1e-8)
        del model
        torch.cuda.empty_cache()
        return pd.DataFrame(pred, columns=targets)

    test_df = pd.read_csv(args.data_path / "test.csv")
    tokenizer = BertTokenizer(
        args.model_dir / "stackx-base-cased-vocab.txt", do_lower_case=False
    )
    checkpoints = list(args.model_dir.glob("*.pth"))

    test_dataset = TestQuestDataset(test_df, tokenizer, max_seg_length=512)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    submission = predict_test_checkpoints(checkpoints, test_loader, ALL_TARGETS)
    submission.to_csv(args.sub_file, index=False)

import os
import argparse
import logging

from mag.experiment import Experiment
import mag
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model import get_model_optimizer
from loops import train_loop, evaluate, infer
from dataset import (
    cross_validation_split,
    get_test_set,
    BucketingSampler,
    make_collate_fn,
)
from transformers import BertTokenizer, AlbertTokenizer
from torch.utils.data import DataLoader, Dataset
from evaluation import target_metric
from misc import target_columns, input_columns

mag.use_custom_separator("-")

parser = argparse.ArgumentParser()

parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--epochs", type=int, nargs="+", required=True)
parser.add_argument("--data_path", type=str, required=True)

args = parser.parse_args()

experiment = Experiment(resume_from=args.experiment)
config = experiment.config

logging.getLogger("transformers").setLevel(logging.ERROR)

train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
submission = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))

original_args = argparse.Namespace(
    folds=config.folds,
    lr=config.lr,
    batch_size=config.batch_size,
    seed=config._seed,
    bert_model=config._bert_model,
    num_classes=30,
    target_columns=target_columns,
    input_columns=input_columns,
    # old models didn't have those parameters in their configs
    max_sequence_length=getattr(config, "max_sequence_length", 500),
    max_title_length=getattr(config, "max_title_length", 26),
    max_question_length=getattr(config, "max_question_length", 260),
    max_answer_length=getattr(config, "max_answer_length", 210),
    head_tail=getattr(config, "head_tail", True),
    use_folds=None,
)

tokenizer = BertTokenizer.from_pretrained(
    original_args.bert_model, do_lower_case=("uncased" in original_args.bert_model),
)

test_set = get_test_set(original_args, test_df, tokenizer)
test_loader = DataLoader(
    test_set,
    batch_sampler=BucketingSampler(
        test_set.lengths,
        batch_size=original_args.batch_size,
        maxlen=original_args.max_sequence_length,
    ),
    collate_fn=make_collate_fn(),
)

val_dfs = []

for (fold, train_set, valid_set, train_fold_df, val_fold_df,) in cross_validation_split(
    original_args, train_df, tokenizer, ignore_train=True
):

    print()
    print("Fold:", fold)
    print()

    valid_loader = DataLoader(
        valid_set,
        batch_sampler=BucketingSampler(
            valid_set.lengths,
            batch_size=original_args.batch_size,
            maxlen=original_args.max_sequence_length,
        ),
        collate_fn=make_collate_fn(),
    )

    fold_checkpoints = os.path.join(experiment.checkpoints, "fold{}".format(fold))
    fold_predictions = os.path.join(experiment.predictions, "fold{}".format(fold))

    criterion = nn.BCEWithLogitsLoss()
    model, optimizer = get_model_optimizer(original_args)

    checkpoint = os.path.join(fold_checkpoints, "model_on_epoch_{}.pth")

    state_dicts = [torch.load(checkpoint.format(epoch)) for epoch in args.epochs]
    averaged_state_dict = state_dicts[0]
    for k in averaged_state_dict:
        averaged_state_dict[k] = torch.mean(
            torch.stack([state_dict[k] for state_dict in state_dicts], dim=0), dim=0,
        )

    torch.save(
        averaged_state_dict,
        os.path.join(
            fold_checkpoints, "swa_{}.pth".format("_".join(map(str, args.epochs))),
        ),
    )

    model.load_state_dict(averaged_state_dict)

    # del state_dicts
    del averaged_state_dict
    torch.cuda.empty_cache()

    avg_val_loss, score, val_preds = evaluate(
        original_args, model, valid_loader, criterion, val_shape=len(valid_set)
    )

    print("Fold {} score: {}".format(fold, score))

    val_preds_df = val_fold_df.copy()[["qa_id"] + target_columns]
    val_preds_df[target_columns] = val_preds
    val_preds_df.to_csv(
        os.path.join(
            fold_predictions, "val_swa_{}.csv".format("_".join(map(str, args.epochs))),
        ),
        index=False,
    )

    torch.cuda.empty_cache()

    val_dfs.append(val_preds_df)

    test_preds = infer(original_args, model, test_loader, test_shape=len(test_set))
    test_preds_df = submission.copy()
    test_preds_df[target_columns] = test_preds
    test_preds_df.to_csv(
        os.path.join(
            fold_predictions, "test_swa_{}.csv".format("_".join(map(str, args.epochs))),
        ),
        index=False,
    )

    torch.cuda.empty_cache()

    print()


oof_df = pd.concat(val_dfs).reset_index(drop=True)
oof_df.to_csv(
    os.path.join(
        experiment.predictions,
        "oof_swa_{}.csv".format("_".join(map(str, args.epochs))),
    ),
    index=False,
)

print("Final metric:", target_metric(oof_df, train_df))

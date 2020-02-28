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
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dataframe", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)

args = parser.parse_args()

experiment = Experiment(resume_from=args.experiment)
config = experiment.config

logging.getLogger("transformers").setLevel(logging.ERROR)

test_df = pd.read_csv(args.dataframe)

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
    original_args.bert_model, do_lower_case=("uncased" in original_args.bert_model)
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

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for fold in range(config.folds):

    print()
    print("Fold:", fold)
    print()

    fold_checkpoints = os.path.join(experiment.checkpoints, "fold{}".format(fold))

    model, optimizer = get_model_optimizer(original_args)

    checkpoint = os.path.join(fold_checkpoints, args.checkpoint)

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    del state_dict
    torch.cuda.empty_cache()

    test_preds = infer(original_args, model, test_loader, test_shape=len(test_set))

    del model, optimizer
    torch.cuda.empty_cache()

    test_preds_df = test_df[["id", "host"]].copy()
    for k, col in enumerate(target_columns):
        test_preds_df[col] = test_preds[:, k].astype(np.float32)
    test_preds_df.to_csv(
        os.path.join(args.output_dir, "fold-{}.csv".format(fold)), index=False,
    )

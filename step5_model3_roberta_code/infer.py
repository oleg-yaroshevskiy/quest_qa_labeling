import argparse
import logging
import os
import mag
import numpy as np
import pandas as pd
import torch
from dataset import get_test_set
from loops import infer
from mag.experiment import Experiment
from misc import target_columns, input_columns
from model import get_model_optimizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer

mag.use_custom_separator("-")

parser = argparse.ArgumentParser()

parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--bert_model", type=str, required=True)
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
    model_type=config.model_type,
)

if original_args.model_type == "bert":
    tokenizer = BertTokenizer.from_pretrained(
        original_args.bert_model, do_lower_case=("uncased" in args.bert_model)
    )
elif original_args.model_type == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained(original_args.bert_model)

test_set = get_test_set(original_args, test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=original_args.batch_size, shuffle=False)

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

    test_preds_df = test_df[["qa_id"]].copy()
    for k, col in enumerate(target_columns):
        test_preds_df[col] = test_preds[:, k].astype(np.float32)
    test_preds_df.to_csv(
        os.path.join(args.output_dir, "fold-{}.csv".format(fold)), index=False,
    )

    torch.cuda.empty_cache()

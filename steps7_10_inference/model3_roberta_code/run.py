import csv
import logging
import os
import random
import warnings

import mag
import numpy as np
import pandas as pd
import torch
from args import args
from dataset import cross_validation_split, get_test_set, get_pseudo_set
from loops import train_loop, evaluate, infer
from mag.experiment import Experiment
from model import get_model_optimizer
from torch import nn
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer

mag.use_custom_separator("-")
warnings.filterwarnings("ignore")

config = {
    "_seed": args.seed,
    "bert_model": args.bert_model.replace("-", "_"),
    "batch_accumulation": args.batch_accumulation,
    "batch_size": args.batch_size,
    "warmup": args.warmup,
    "lr": args.lr,
    "folds": args.folds,
    "max_sequence_length": args.max_sequence_length,
    "max_title_length": args.max_title_length,
    "max_question_length": args.max_question_length,
    "max_answer_length": args.max_answer_length,
    "head_tail": args.head_tail,
    "label": args.label,
    "_pseudo_file": args.pseudo_file,
    "model_type": args.model_type,
}
experiment = Experiment(config, implicit_resuming=args.use_folds is not None)
experiment.register_directory("checkpoints")
experiment.register_directory("predictions")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


logging.getLogger("transformers").setLevel(logging.ERROR)
seed_everything(args.seed)
# load the data
train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
submission = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))

if args.model_type == "bert":
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=("uncased" in args.bert_model)
    )
elif args.model_type == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)

test_set = get_test_set(args, test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

for fold, train_set, valid_set, train_fold_df, val_fold_df in cross_validation_split(
    args, train_df, tokenizer
):

    print()
    print("Fold:", fold)
    print()

    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    fold_checkpoints = os.path.join(experiment.checkpoints, "fold{}".format(fold))
    fold_predictions = os.path.join(experiment.predictions, "fold{}".format(fold))

    os.makedirs(fold_checkpoints, exist_ok=True)
    os.makedirs(fold_predictions, exist_ok=True)

    log_file = os.path.join(fold_checkpoints, "training_log.csv")

    with open(log_file, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "loss", "val_loss", "score"])

    iteration = 0
    best_score = -1.0

    model, optimizer = get_model_optimizer(args)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):

        epoch_train_set = train_set

        if args.pseudo_file is not None:
            pseudo_df = pd.read_csv(args.pseudo_file.format(fold))

            pseudo_set = get_pseudo_set(
                args, pseudo_df.sample(args.n_pseudo), tokenizer
            )

            epoch_train_set = ConcatDataset([epoch_train_set, pseudo_set])

        train_loader = DataLoader(
            epoch_train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
            shuffle=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup,
            num_training_steps=(
                args.epochs * len(train_loader) / args.batch_accumulation
            ),
        )

        avg_loss, iteration = train_loop(
            model, train_loader, optimizer, criterion, scheduler, args, iteration,
        )

        avg_val_loss, score, val_preds = evaluate(
            args, model, valid_loader, criterion, val_shape=len(valid_set)
        )

        print(
            "Epoch {}/{}: \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f}".format(
                epoch + 1, args.epochs, avg_loss, avg_val_loss, score
            )
        )
        with open(log_file, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, avg_loss, avg_val_loss, score])

        torch.save(
            model.state_dict(),
            os.path.join(fold_checkpoints, "model_on_epoch_{}.pth".format(epoch)),
        )
        val_preds_df = val_fold_df.copy()[["qa_id"] + args.target_columns]
        val_preds_df[args.target_columns] = val_preds
        val_preds_df.to_csv(
            os.path.join(fold_predictions, "val_on_epoch_{}.csv".format(epoch)),
            index=False,
        )

        test_preds = infer(args, model, test_loader, test_shape=len(test_set))
        test_preds_df = submission.copy()
        test_preds_df[args.target_columns] = test_preds
        test_preds_df.to_csv(
            os.path.join(fold_predictions, "test_on_epoch_{}.csv".format(epoch)),
            index=False,
        )

        if score > best_score:
            best_score = score
            torch.save(
                model.state_dict(), os.path.join(fold_checkpoints, "best_model.pth"),
            )
            val_preds_df.to_csv(
                os.path.join(fold_predictions, "best_val.csv"), index=False
            )
            test_preds_df.to_csv(
                os.path.join(fold_predictions, "best_test.csv"), index=False
            )

    torch.cuda.empty_cache()

    print()

best_val_df_files = [
    os.path.join(experiment.predictions, "fold{}".format(fold), "best_val.csv")
    for fold in range(args.folds)
]

if all(os.path.isfile(file) for file in best_val_df_files):
    best_val_dfs = [pd.read_csv(file) for file in best_val_df_files]
    # create oof df only if all folds were used
    oof_df = pd.concat(best_val_dfs).reset_index(drop=True)
    oof_df.to_csv(os.path.join(experiment.predictions, "oof.csv"), index=False)

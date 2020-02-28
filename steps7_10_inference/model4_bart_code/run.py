import random
import os, multiprocessing, glob
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from fairseq.models.bart import BARTModel
from model import get_model
from loops import infer
from dataset import get_test_set
from args import args
from torch.utils.data import DataLoader, Dataset

PATH_TO_BART_CKPT = "input/model4_ckpt/"

test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
submission = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))
submission[args.target_columns] = 0.0

tokenizer = BARTModel.from_pretrained(args.bert_model, include_model=False)

test_set = get_test_set(args, test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = get_model(args)

for fold in range(args.folds):
    print("Fold", fold)
    model.load_state_dict(
        torch.load("{}/fold{}/best_model.pth".format(PATH_TO_BART_CKPT, fold)),
        strict=False,
    )
    test_preds = infer(args, model, test_loader, test_shape=len(test_set))
    submission[args.target_columns] += test_preds

submission[args.target_columns] /= 5
submission.to_csv(args.sub_file, index=False)

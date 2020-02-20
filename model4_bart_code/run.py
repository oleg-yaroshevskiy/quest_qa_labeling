import os
import pandas as pd
import torch
from args import args
from dataset import get_test_set
from fairseq.models.bart import BARTModel
from loops import infer
from model import get_model
from torch.utils.data import DataLoader

test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
submission = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))
submission[args.target_columns] = 0.

tokenizer = BARTModel.from_pretrained(args.bert_model, include_model=False)

test_set = get_test_set(args, test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = get_model(args)

for fold in range(args.folds):
    print("Fold", fold)
    model.load_state_dict(torch.load("/kaggle/input/quest-bart/fold{}/best_model.pth".format(fold)), strict=False)
    test_preds = infer(args, model, test_loader, test_shape=len(test_set))
    submission[args.target_columns] += test_preds

submission[args.target_columns] /= 5
submission.to_csv(args.sub_file, index=False)

import warnings
import os
import pandas as pd
import torch
from model import get_model_optimizer
from loops import infer
from dataset import get_test_set
from args import args
from transformers import BertTokenizer
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
submission = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))
submission[args.target_columns] = 0.0

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
test_set = get_test_set(args, test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = get_model_optimizer(args)

for fold in range(args.folds):
    print("Fold", fold)
    model.load_state_dict(
        torch.load("{}/fold{}/best_model.pth".format(args.checkpoints, fold))
    )
    test_preds = infer(args, model, test_loader, test_shape=len(test_set))
    submission[args.target_columns] += test_preds

submission[args.target_columns] /= args.folds
submission.to_csv(args.sub_file, index=False)

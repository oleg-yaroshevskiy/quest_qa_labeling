import os
import sys
from pathlib import Path

import pandas as pd
import torch
from callbacks import CSVParamLogger
from data import QuestDataset, ALL_TARGETS
from metrics import Spearman
from models import BertForQuestRegression
from poutyne.framework import Model, ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

if len(sys.argv) == 1 or sys.argv[1] != "toy":
    train_df = pd.read_csv("input/google-quest-challenge/train.csv")
    N_EPOCHS, N_FOLDS = 8, 5
else:
    train_df = pd.read_csv("input/google-quest-challenge/train_toy.csv")
    N_EPOCHS, N_FOLDS = 1, 1

checkpoint_dir = Path("input/stackx-base-cased/")
out_checkpoint_dir = Path("input/model1_ckpt/")

os.makedirs(str(checkpoint_dir), exist_ok=True)

tokenizer = BertTokenizer(
    str(checkpoint_dir / "stackx-base-cased-vocab.txt"), do_lower_case=False
)


def get_model():
    config = BertConfig.from_json_file(
        str(checkpoint_dir / "stackx-base-cased-config.json")
    )
    config.__dict__["num_labels"] = len(ALL_TARGETS)

    model = BertForQuestRegression(config)
    state_dict = torch.load(
        checkpoint_dir
        / "stackx_base_with_aux_ep_080_val_perplexity_4.39_val_spearman_0.34.pth",
        map_location="cpu",
    )
    state_dict = {
        name.replace("LayerNorm.gamma", "LayerNorm.weight").replace(
            "LayerNorm.beta", "LayerNorm.bias"
        ): weight
        for name, weight in state_dict.items()
    }

    del state_dict["classifier.weight"]
    del state_dict["classifier.bias"]
    model.load_state_dict(state_dict, strict=False)
    return model


folds = pd.read_csv("input/model1_folds.txt")["fold"]

loader_kws = dict(batch_size=4, num_workers=1)

spearman = Spearman(ALL_TARGETS)
for fold_idx in range(N_FOLDS):
    train_frame, val_frame = train_df[folds != fold_idx], train_df[folds == fold_idx]

    train_dataset = QuestDataset(train_frame, tokenizer, max_seg_length=512)
    val_dataset = QuestDataset(val_frame, tokenizer, max_seg_length=512)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kws)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kws)

    model = get_model()
    trainer = Model(
        model,
        Adam(model.parameters(), lr=3e-5),
        "bce_with_logits",
        epoch_metrics=[spearman],
    )
    trainer.to("cuda")

    checkpoint_name = (
        "stackx_with_aux_80_fold_"
        + str(fold_idx)
        + "_ep_{epoch:03}_val_spearman_{val_spearman:.2f}.pth"
    )
    callbacks = [
        spearman.callback,
        ModelCheckpoint(str(out_checkpoint_dir / checkpoint_name), atomic_write=False),
        CSVParamLogger(
            str(out_checkpoint_dir / "training_log.csv"),
            append=fold_idx != 0,
            extra_metrics=ALL_TARGETS,
        ),
    ]

    history = trainer.fit_generator(
        train_loader, val_loader, epochs=N_EPOCHS, callbacks=callbacks,
    )
    model.to("cpu")
    del model
    torch.cuda.empty_cache()

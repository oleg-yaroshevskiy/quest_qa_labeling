import numpy as np
import torch
import gc
from scipy.stats import spearmanr
from tqdm import tqdm


def train_loop(model, train_loader, optimizer, criterion, scheduler, args, iteration):
    model.train()

    avg_loss = 0.0

    optimizer.zero_grad()
    for idx, batch in enumerate(tqdm(train_loader, desc="Train", ncols=80)):
        input_ids, input_masks, input_segments, labels = (
            batch["input_ids"],
            batch["input_masks"],
            batch["input_segments"],
            batch["labels"],
        )
        input_ids, input_masks, input_segments, labels = (
            input_ids.cuda(),
            input_masks.cuda(),
            input_segments.cuda(),
            labels.cuda(),
        )

        logits = model(
            input_ids=input_ids.long(),
            attention_mask=input_masks,
            token_type_ids=input_segments,
        )
        loss = criterion(logits, labels)

        loss.backward()
        if (iteration + 1) % args.batch_accumulation == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        iteration += 1

        avg_loss += loss.item() / (len(train_loader) * args.batch_accumulation)

    torch.cuda.empty_cache()
    gc.collect()
    return (avg_loss, iteration)


def evaluate(args, model, val_loader, criterion, val_shape):
    avg_val_loss = 0.0
    model.eval()

    valid_preds = []
    original = []
    ids = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Valid", ncols=80)):
            id, input_ids, input_masks, input_segments, labels = (
                batch["idx"],
                batch["input_ids"],
                batch["input_masks"],
                batch["input_segments"],
                batch["labels"],
            )
            ids.extend(id.cpu().numpy())
            input_ids, input_masks, input_segments, labels = (
                input_ids.cuda(),
                input_masks.cuda(),
                input_segments.cuda(),
                labels.cuda(),
            )

            logits = model(
                input_ids=input_ids.long(),
                attention_mask=input_masks,
                token_type_ids=input_segments,
            )

            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds.extend(logits.detach().cpu().numpy())
            original.extend(labels.detach().cpu().numpy())

        valid_preds = np.array(valid_preds)
        original = np.array(original)

        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()

        for i in range(len(args.target_columns)):
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)

    return avg_val_loss, score / len(args.target_columns), preds[np.argsort(ids)]


def infer(args, model, test_loader, test_shape):
    test_preds = np.zeros((test_shape, args.num_classes))
    model.eval()

    ids = []
    test_preds = []

    for idx, batch in enumerate(tqdm(test_loader, desc="Test", ncols=80)):
        with torch.no_grad():

            ids.extend(batch["idx"].cpu().numpy())

            predictions = model(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["input_masks"].cuda(),
                token_type_ids=batch["input_segments"].cuda(),
            )
            test_preds.extend(predictions.detach().cpu().numpy())

    test_preds = np.array(test_preds)

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output[np.argsort(ids)]

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
        input_ids, labels, _ = batch
        input_ids, labels = (input_ids.cuda(), labels.cuda())

        logits = model(input_ids=input_ids.long())
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

    valid_preds = np.zeros((val_shape, args.num_classes))
    original = np.zeros((val_shape, args.num_classes))

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Valid", ncols=80)):
            input_ids, labels, _ = batch
            input_ids, labels = (
                input_ids.cuda(),
                labels.cuda(),
            )

            logits = model(input_ids=input_ids.long())

            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx * args.batch_size : (idx + 1) * args.batch_size] = (
                logits.detach().cpu().squeeze().numpy()
            )
            original[idx * args.batch_size : (idx + 1) * args.batch_size] = (
                labels.detach().cpu().squeeze().numpy()
            )

        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()

        for i in range(len(args.target_columns)):
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)

    return avg_val_loss, score / len(args.target_columns), preds


def infer(args, model, test_loader, test_shape):
    test_preds = np.zeros((test_shape, args.num_classes))
    model.eval()

    for idx, x_batch in enumerate(tqdm(test_loader, desc="Test", ncols=80)):
        with torch.no_grad():
            predictions = model(input_ids=x_batch[0].cuda())
            test_preds[idx * args.batch_size : (idx + 1) * args.batch_size] = (
                predictions.detach().cpu().squeeze().numpy()
            )

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import torch_to_numpy, torch_to


def predict(model, test_loader, columns):
    model.eval()
    model.cuda()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            pred = torch_to_numpy(model(*torch_to(batch, 'cuda')))
            preds.append(pred)

    preds = np.vstack(preds)

    preds = torch.sigmoid(torch.from_numpy(preds)).numpy()
    preds = np.clip(preds, 0, 1 - 1e-8)
    preds = pd.DataFrame(preds, columns=columns)
    return preds

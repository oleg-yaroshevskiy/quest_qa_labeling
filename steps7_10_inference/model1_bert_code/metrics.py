import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from poutyne.framework.metrics import EpochMetric
from poutyne.framework.callbacks import Callback
from poutyne.utils import torch_to_numpy


def spearman_metric(y_true, y_pred, return_scores=False, colnames=None):
    corr = [
        spearmanr(pred_col, target_col).correlation
        for pred_col, target_col in zip(y_pred.T, y_true.T)
    ]
    if colnames is not None:
        return pd.Series(corr, index=colnames)
    if return_scores:
        return corr
    else:
        return np.nanmean(corr)


class Spearman(EpochMetric):
    class SpearmanCallback(Callback):
        def __init__(self):
            self.metric_values = dict()

        def on_epoch_end(self, epoch, logs):
            logs.update(self.metric_values)

    def __init__(self, colnames=None):
        super(Spearman, self).__init__()
        self.__name__ = "spearman"
        self.preds, self.targets = [], []

        self.colnames = colnames
        self.callback = self.SpearmanCallback()

    def forward(self, y_pred, y_true):
        self.preds.append(torch_to_numpy(y_pred))
        self.targets.append(torch_to_numpy(y_true))

    def get_metric(self):
        corr = spearman_metric(
            np.vstack(self.targets), np.vstack(self.preds), return_scores=True
        )

        if self.colnames is not None:
            self.callback.metric_values = dict(zip(self.colnames, corr))

        self.preds, self.targets = [], []
        return np.mean(corr)

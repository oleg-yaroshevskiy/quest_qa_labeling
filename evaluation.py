import pandas as pd
from scipy.stats import spearmanr
import numpy as np

from misc import target_columns


def target_metric(prediction, actual):

    prediction = prediction.sort_values(by="qa_id").reset_index(drop=True)
    actual = actual.sort_values(by="qa_id").reset_index(drop=True)

    assert (prediction.qa_id == actual.qa_id).all()

    score = 0
    for col in target_columns:
        score += np.nan_to_num(
            spearmanr(prediction[col], actual[col]).correlation
        ) / len(target_columns)
    return score

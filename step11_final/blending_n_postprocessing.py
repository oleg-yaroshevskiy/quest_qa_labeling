from collections import Counter
import numpy as np
import pandas as pd
from pathlib2 import Path

PATH_TO_DATA = Path("input/google-quest-challenge/")
PATH_TO_SUBMISSIONS = Path("submissions")

# Reading the 30 target columns that we need to predict

sample_submission_df = pd.read_csv(
    PATH_TO_DATA / "sample_submission.csv", index_col="qa_id"
)
target_columns = sample_submission_df.columns

train_df = pd.read_csv(PATH_TO_DATA / "train.csv")

# Reading submission files

model1_pred_df = pd.read_csv(PATH_TO_SUBMISSIONS / "model1_submission.csv")
model2_pred_df = pd.read_csv(PATH_TO_SUBMISSIONS / "model2_bert_base_cased_pred.csv")
model4_pred_df = pd.read_csv(PATH_TO_SUBMISSIONS / "model4_bart_large_pred.csv")

# For RoBERTa, we average predictions from 5 folds

roberta_base_dfs = [
    pd.read_csv(
        PATH_TO_SUBMISSIONS / "model3_roberta-base-output" / "fold-{}.csv".format(fold)
    )
    for fold in range(5)
]

model3_pred_df = roberta_base_dfs[0].copy()

for col in target_columns:
    model3_pred_df[col] = np.mean([df[col] for df in roberta_base_dfs], axis=0)

# Blending

blended_df = model3_pred_df.copy()

for col in target_columns:
    blended_df[col] = (
        model1_pred_df[col] * 0.1
        + model2_pred_df[col] * 0.2
        + model3_pred_df[col] * 0.1
        + model4_pred_df[col] * 0.3
    )


# Applying postprocessing to the final blend
def postprocess_single(target, ref):
    """
    The idea here is to make the distribution of a particular predicted column
    to match the correspoding distribution of the corresponding column in the
    training dataset (called ref here)
    """

    ids = np.argsort(target)
    counts = sorted(Counter(ref).items(), key=lambda s: s[0])
    scores = np.zeros_like(target)

    last_pos = 0
    v = 0

    for value, count in counts:
        next_pos = last_pos + int(round(count / len(ref) * len(target)))
        if next_pos == last_pos:
            next_pos += 1

        cond = ids[last_pos:next_pos]
        scores[cond] = v
        last_pos = next_pos
        v += 1

    return scores / scores.max()


def postprocess_prediction(prediction, actual):
    postprocessed = prediction.copy()

    for col in target_columns:
        scores = postprocess_single(prediction[col].values, actual[col].values)
        # Those are columns where our postprocessing gave substantial improvement.
        # It also helped for some others, but we didn't include them as the gain was
        # very marginal (less than 0.01)
        if col in (
            "question_conversational",
            "question_type_compare",
            "question_type_definition",
            "question_type_entity",
            "question_has_commonly_accepted_answer",
            "question_type_consequence",
            "question_type_spelling",
        ):
            postprocessed[col] = scores

        # scale to 0-1 interval
        v = postprocessed[col].values
        postprocessed[col] = (v - v.min()) / (v.max() - v.min())

    return postprocessed


postprocessed = postprocess_prediction(blended_df, train_df)

# Saving the submission file

postprocessed.to_csv(PATH_TO_SUBMISSIONS / "submission.csv", index=False)

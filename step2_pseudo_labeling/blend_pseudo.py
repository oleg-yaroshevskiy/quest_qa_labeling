import os
import sys
import numpy as np
import pandas as pd

if len(sys.argv) == 1 or sys.argv[1] != "toy":
    original_df = pd.read_csv("input/sampled_sx_so.csv.gz")
else:
    original_df = pd.read_csv("input/qa_stackexchange_cleaned_toy.csv")

bert_base_dfs = [
    pd.read_csv("pseudo-predictions/base/fold-{}.csv".format(fold)) for fold in range(5)
]
bert_large_dfs = [
    pd.read_csv("pseudo-predictions/large/fold-{}.csv".format(fold))
    for fold in range(5)
]
bert_base_pretrained_dfs = [
    pd.read_csv("pseudo-predictions/base-pretrained/fold-{}.csv".format(fold))
    for fold in range(5)
]

target_columns = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

os.makedirs(
    os.path.join("pseudo-predictions", "pseudo-100k-3x-blend-no-leak"), exist_ok=True
)

for fold in range(5):

    pseudo_df = bert_base_dfs[0].copy()

    for col in target_columns:
        blended = (
            bert_base_dfs[fold][col] * 0.2
            + bert_large_dfs[fold][col] * 0.4
            + bert_base_pretrained_dfs[fold][col] * 0.4
        ).astype(np.float16)

        pseudo_df[col] = blended

    final_df = pd.concat([original_df, pseudo_df], axis=1)

    final_df.to_csv(
        os.path.join(
            "pseudo-predictions",
            "pseudo-100k-3x-blend-no-leak",
            "fold-{}.csv.gz".format(fold),
        ),
        index=False,
    )

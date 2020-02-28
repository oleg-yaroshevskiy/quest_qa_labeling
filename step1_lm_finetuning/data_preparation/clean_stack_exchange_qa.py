import re
from multiprocessing import Pool
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


def merge_all_questions_and_answers(path_to_parsed_dumps):
    dumps = path_to_parsed_dumps.glob("*")
    dumps = [path for path in dumps if path.joinpath("questions.tsv").exists()]

    all_questions, all_answers = [], []
    offset = 0

    for dump in tqdm(dumps):
        questions = pd.read_csv(dump / "questions.tsv", sep="\t")
        answers = pd.read_csv(dump / "answers.tsv", sep="\t")

        questions["host"] = answers["host"] = dump.name

        accepted_answers = questions["AcceptedAnswerId"].dropna().astype(int)
        answers["is_answer_accepted"] = answers["Id"].astype(int).isin(accepted_answers)

        questions["Id"] = questions["Id"].astype(int) + offset
        answers["Id"] = answers["Id"].astype(int) + offset
        answers["ParentId"] = answers["ParentId"].astype(int) + offset

        all_questions.append(questions)
        all_answers.append(answers)

        offset = max(questions["Id"].max(), answers["Id"].max()) + 1

    all_questions = pd.concat(all_questions, sort=False).reset_index()
    all_answers = pd.concat(all_answers, sort=False).reset_index()

    return all_questions, all_answers


def process(
    all_questions,
    all_answers,
    path_to_save,
    detect_lang=False,
    n_jobs=4,
    chunksize=100000,
):
    html_pattern = re.compile(r"<.*?>")

    question_body = (
        all_questions["Body"]
        .astype(str)
        .progress_apply(lambda s: html_pattern.sub("", s))
    )
    question_title = (
        all_questions["Title"]
        .astype(str)
        .progress_apply(lambda s: html_pattern.sub("", s))
    )
    question_body_cleaned = question_body.progress_apply(
        lambda s: s.replace("\n", "")
    ).values

    question_ans_agg_features = pd.concat(
        [
            all_answers.groupby("ParentId")["Score"].max().rename("answers_max_score"),
            all_answers.groupby("ParentId")["Score"]
            .mean()
            .rename("answers_mean_score"),
        ],
        axis=1,
        sort=False,
    )

    question_features = all_questions[
        ["Id", "host", "username", "Score", "ViewCount", "FavoriteCount", "AnswerCount"]
    ]
    question_features = question_features.merge(
        question_ans_agg_features, left_on="Id", right_on="ParentId", how="left"
    )
    question_features = pd.concat(
        [question_features, question_title, question_body], axis=1, sort=False
    )

    if detect_lang:
        n_chunks = int(len(question_body_cleaned) / chunksize) + 1

        def predict_lang(sentences):
            model = fasttext.load_model("lid.176.bin")
            return model.predict(list(sentences), k=1)[0]

        with Pool(n_jobs) as pool:
            question_lang = list(
                tqdm(
                    pool.imap(
                        predict_lang,
                        [
                            question_body_cleaned[i * chunksize : (i + 1) * chunksize]
                            for i in range(n_chunks)
                        ],
                    ),
                    total=n_chunks,
                )
            )
        question_lang = list(map(np.array, question_lang))
        question_lang = np.vstack(question_lang).flatten()

        question_features = question_features.iloc[
            (question_lang == "en") & (all_questions["AnswerCount"].ravel() != 0)
        ]
    else:
        question_features = question_features.iloc[
            (all_questions["AnswerCount"].ravel() != 0)
        ]

    question_features.columns = [
        "id",
        "host",
        "question_username",
        "question_score",
        "question_views",
        "question_favs",
        "answers_count",
        "answers_max_score",
        "answers_mean_score",
        "title",
        "body",
    ]
    selected_answers = all_answers["Id"].isin(select_answers(all_answers))

    answer_features = all_answers[selected_answers][
        ["ParentId", "username", "Body", "Score", "is_answer_accepted"]
    ]
    answer_features["Body"] = (
        answer_features["Body"]
        .astype(str)
        .progress_apply(lambda s: html_pattern.sub("", s))
    )

    answer_features.columns = [
        "ParentId",
        "answer_username",
        "answer",
        "answer_score",
        "is_answer_accepted",
    ]
    qa_features = question_features.merge(
        answer_features, left_on="id", right_on="ParentId", how="inner"
    ).drop(columns=["ParentId"])

    targets = [
        "question_score",
        "question_views",
        "question_favs",
        "answers_count",
        "answers_max_score",
        "answers_mean_score",
        "answer_score",
        "is_answer_accepted",
    ]

    qa_features[targets] = qa_features[targets].fillna(0)
    qa_features[targets] = np.sign(qa_features[targets]) * np.log1p(
        np.abs(qa_features[targets].astype("float"))
    )
    qa_features[targets] = qa_features[targets] / qa_features[targets].std()

    # save results
    qa_features.rename(
        columns={"body": "question_body", "title": "question_title"}
    ).to_csv(path_to_save / "qa_stackexchange_cleaned.csv", index=False)
    qa_features.rename(
        columns={"body": "question_body", "title": "question_title"}
    ).head(50).to_csv(path_to_save / "qa_stackexchange_cleaned_toy.csv", index=False)


def select_answers(all_answers, max_answers_per_question=2):
    answer_ids = all_answers[["Id", "ParentId", "is_answer_accepted"]]

    selected_answers = answer_ids["Id"][answer_ids["is_answer_accepted"]].tolist()
    n_additional = max_answers_per_question - (
        answer_ids.groupby("ParentId")["is_answer_accepted"].sum() > 0
    ).astype(int)

    additional_answers = answer_ids.merge(
        n_additional.rename("n_additional").reset_index(), on="ParentId"
    )

    while len(additional_answers) > 0:
        print(len(additional_answers))
        additional_answers = additional_answers[
            ~additional_answers["Id"].isin(selected_answers)
        ]
        additional_answers = additional_answers[additional_answers["n_additional"] > 0]
        additional_answers = additional_answers.iloc[
            np.random.permutation(len(additional_answers))
        ].reset_index(drop=True)
        selected_answers += (
            additional_answers["Id"]
            .loc[additional_answers["ParentId"].drop_duplicates().index]
            .tolist()
        )

        additional_answers["n_additional"] = additional_answers["n_additional"] - 1
    return selected_answers


if __name__ == "__main__":
    PATH_TO_SX_PARSED = Path("input/sx_dump/stackexchange_parsed")
    PATH_TO_SAVE_RESULT = Path("input")

    # if this flag is on, then we detect question language with fasttext
    # and leave only english
    DETECT_LANG = False

    # merge all parsed questions and answers output by `scrape_stack_exchange.py`
    all_questions, all_answers = merge_all_questions_and_answers(
        path_to_parsed_dumps=PATH_TO_SX_PARSED
    )

    process(
        all_questions,
        all_answers,
        path_to_save=PATH_TO_SAVE_RESULT,
        detect_lang=DETECT_LANG,
    )

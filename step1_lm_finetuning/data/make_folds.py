import pandas as pd
import functools
import operator
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import IterativeStratification
from multiprocessing import Pool, cpu_count

from utils import transform_target_columns_to_ordinals


def rareness_split(train_df, least_representative_cols=("question_type_spelling",)):
    rarity_mask = []
    for col in least_representative_cols:
        most_common_value = (
            train_df[col].value_counts().sort_values(ascending=False).index[0]
        )
        rarity_mask.append(train_df[col] != most_common_value)
    rarity_mask = functools.reduce(operator.or_, rarity_mask)

    rare_samples = train_df[rarity_mask]
    common_samples = train_df[~rarity_mask]

    return common_samples, rare_samples


def aggregate_ordinals(group, agg_func=pd.Series.mode):
    group_id, group_labels = group
    group_labels.drop(labels=["group_id"], axis=1, inplace=True)

    agg_group_labels = group_labels.apply(agg_func, axis=0).iloc[0]
    agg_group_labels = agg_group_labels.rename(group_id)
    return agg_group_labels


def stratified_fold_split_for_common(
    common_samples,
    n_splits=5,
    interaction_order=1,
    random_state=42,
    agg_func=pd.Series.mode,
):
    body_encoder = LabelEncoder()
    common_samples["group_id"] = body_encoder.fit_transform(
        common_samples["question_body"].astype(str)
    )

    common_ordinals = transform_target_columns_to_ordinals(common_samples)
    common_ordinals["group_id"] = common_samples["group_id"]

    common_groups = common_ordinals.groupby(["group_id"])
    with Pool(cpu_count()) as pool:
        aggregated_common_ordinals = list(
            tqdm(
                pool.imap(
                    functools.partial(aggregate_ordinals, agg_func=agg_func),
                    common_groups,
                ),
                total=len(common_groups),
                desc="Aggregate ordinals over groups",
            )
        )
    aggregated_common_ordinals = pd.concat(
        aggregated_common_ordinals, axis=1
    ).transpose()
    aggregated_common_ordinals.index.rename("group_id", inplace=True)

    k_fold = IterativeStratification(
        n_splits=n_splits, order=interaction_order, random_state=random_state
    )
    folds_common = []
    for _, fold_groups in k_fold.split(
        aggregated_common_ordinals, aggregated_common_ordinals
    ):
        fold_mask = common_ordinals["group_id"].isin(fold_groups)
        fold_ids = common_ordinals.index.values[fold_mask]
        folds_common.append(fold_ids)

    return folds_common


def stratified_fold_split_for_rare(
    rare_samples,
    n_splits=5,
    interaction_order=1,
    random_state=42,
    least_representative_cols=("question_type_spelling",),
):
    rare_ordinals = transform_target_columns_to_ordinals(
        rare_samples[list(least_representative_cols)]
    )

    k_fold = IterativeStratification(
        n_splits=n_splits, order=interaction_order, random_state=random_state
    )
    folds_rare = []
    for _, fold_ids in k_fold.split(rare_ordinals, rare_ordinals):
        folds_rare.append(rare_samples.index.values[fold_ids])
    return folds_rare

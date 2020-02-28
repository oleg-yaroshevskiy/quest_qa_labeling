from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from data.dataset import ALL_TARGETS


def encode_labels(df, target_columns=ALL_TARGETS, method="average"):
    target_columns = [t for t in target_columns if t in df]
    df = deepcopy(df)

    ranked = [rankdata(df[col], method=method).reshape(-1, 1) for col in target_columns]
    ranked = np.hstack(ranked) - 1
    df[target_columns] = ranked

    return df


def transform_target_columns_to_ordinals(
    df, target_columns=ALL_TARGETS, return_label_groups=False
):
    target_columns = [t for t in target_columns if t in df]
    rank_labels = encode_labels(df, target_columns, method="dense")

    ordinal_labels = []
    for col in target_columns:
        col_labels = rank_labels[col]

        new_col_names = [col + "_" + str(i) for i in sorted(np.unique(col_labels))]
        new_col_names = new_col_names[1:]
        ordinals = np.array(
            [[1] * label + [0] * (len(new_col_names) - label) for label in col_labels]
        )

        ordinals = pd.DataFrame(ordinals, columns=new_col_names, index=df.index)
        ordinal_labels.append(ordinals)
    ordinal_labels = pd.concat(ordinal_labels, axis=1)

    if return_label_groups:
        label_groups = {label: [] for label in target_columns}
        for label_idx, label in enumerate(ordinal_labels.columns):
            for group in label_groups:
                if label.startswith(group):
                    label_groups[group].append(label_idx)

        return ordinal_labels, label_groups
    else:
        return ordinal_labels


def torch_to_numpy(obj, copy=False):
    """
    Convert to Numpy arrays all tensors inside a Python object composed of the supported types.

    Args:
        obj: The Python object to convert.
        copy (bool): Whether to copy the memory. By default, if a tensor is already on CPU, the
            Numpy array will be a view of the tensor.

    Returns:
        A new Python object with the same structure as `obj` but where the tensors are now Numpy
        arrays. Not supported type are left as reference in the new object.

    Example:
        .. code-block:: python

            >>> from poutyne import torch_to_numpy
            >>> torch_to_numpy({
            ...     'first': torch.tensor([1, 2, 3]),
            ...     'second':[torch.tensor([4,5,6]), torch.tensor([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': array([1, 2, 3]),
                'second': [array([4, 5, 6]), array([7, 8, 9])],
                'third': 34
            }

    See:
        :meth:`~poutyne.torch_apply` for supported types.
    """
    if copy:
        func = lambda t: t.cpu().detach().numpy().copy()
    else:
        func = lambda t: t.cpu().detach().numpy()
    return torch_apply(obj, func)


def torch_to(obj, *args, **kargs):
    return torch_apply(obj, lambda t: t.to(*args, **kargs))


def torch_apply(obj, func):
    """
    Apply a function to all tensors inside a Python object composed of the supported types.

    Supported types are: list, tuple and dict.

    Args:
        obj: The Python object to convert.
        func: The function to apply.

    Returns:
        A new Python object with the same structure as `obj` but where the tensors have been applied
        the function `func`. Not supported type are left as reference in the new object.
    """
    fn = lambda t: func(t) if torch.is_tensor(t) else t
    return _apply(obj, fn)


def _apply(obj, func):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k: _apply(el, func) for k, el in obj.items()}
    return func(obj)


def _concat(obj):
    if isinstance(obj[0], (list, tuple)):
        return tuple([_concat(ele) for ele in zip(*obj)])
    if isinstance(obj[0], dict):
        concat_dict = {}
        for key in obj[0].keys():
            concat_dict[key] = _concat([o[key] for o in obj])
        return concat_dict
    return np.concatenate(obj)


def numpy_to_torch(obj):
    """
    Convert to tensors all Numpy arrays inside a Python object composed of the supported types.

    Args:
        obj: The Python object to convert.

    Returns:
        A new Python object with the same structure as `obj` but where the Numpy arrays are now
        tensors. Not supported type are left as reference in the new object.

    Example:
        .. code-block:: python

            >>> from poutyne import numpy_to_torch
            >>> numpy_to_torch({
            ...     'first': np.array([1, 2, 3]),
            ...     'second':[np.array([4,5,6]), np.array([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': tensor([1, 2, 3]),
                'second': [tensor([4, 5, 6]), tensor([7, 8, 9])],
                'third': 34
            }


    """
    fn = lambda a: torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    return _apply(obj, fn)

import random
from collections import defaultdict
from typing import Tuple, Iterable, Optional

import numpy as np
import pandas as pd

from .dataset import filtered_and_canonicalized_dataset, dataset_info


def train_val_test_split(dataset: str,
                         return_indices: bool = False,
                         data_list: Optional[Iterable] = None,
                         random_state: Optional[int] = None) -> Tuple[Iterable, Iterable, Iterable]:
    """Interface API for splitting `dataset`.

    Splitting method:
        Random split: Index after shuffling;
        Scaffold split: Deterministic scaffold split.
    """
    df = filtered_and_canonicalized_dataset(dataset)
    if data_list is None:
        data_list = df

    assert len(data_list) == len(df), \
           'Number of samples in `data_list` does not match that in filtered dataset.'

    info = dataset_info(dataset)
    if info.splitting == 'random':
        assert random_state is not None, '`random_state` not provided for random split.'
        return random_split(data_list, return_indices, random_state)
    elif info.splitting == 'scaffold':
        smiles_sr = df.loc[:, info.smiles_column]
        return scaffold_split(smiles_sr, return_indices, data_list)

    raise ValueError(f'Unsupported dataset for splitting: "{dataset}".')


def rigorous_train_val_test_split(dataset: str,
                                  random_state: int,
                                  n_splits: Optional[int] = None,
                                  split_index: Optional[int] = None,
                                  return_indices: bool = False,
                                  data_list: Optional[Iterable] = None) -> Tuple[Iterable, Iterable, Iterable]:
    """Interface API for splitting `dataset` more rigorously.

    Splitting method:
        Random split: One in k-fold split;
        Scaffold split: Randomized scaffold split.
    """
    df = filtered_and_canonicalized_dataset(dataset)
    if data_list is None:
        data_list = df

    assert len(data_list) == len(df), \
           'Number of samples in `data_list` does not match that in filtered dataset.'

    info = dataset_info(dataset)
    if info.splitting == 'random':
        assert n_splits is not None and split_index is not None, \
               '`n_splits` or `split_index` not provided for one in k-fold split.'
        return one_in_k_fold_split(data_list, n_splits, split_index, return_indices, random_state)
    elif info.splitting == 'scaffold':
        smiles_sr = df.loc[:, info.smiles_column]
        return randomized_scaffold_split(smiles_sr, return_indices, data_list, random_state)

    raise ValueError(f'Unsupported dataset for splitting: "{dataset}".')


def random_split(data_list: Iterable,
                 return_indices: bool,
                 random_state: int) -> Tuple[Iterable, Iterable, Iterable]:
    """Random split: Index after shuffling."""
    indices = list(range(len(data_list)))

    random.seed(random_state)
    random.shuffle(indices)

    train_cutoff = int(len(indices) * 0.8)
    val_cutoff = int(len(indices) * (0.8+0.1))
    train_indices, val_indices, test_indices = \
        indices[: train_cutoff], indices[train_cutoff : val_cutoff], indices[val_cutoff :]

    if return_indices:
        return train_indices, val_indices, test_indices

    return _split_by_indices(data_list, train_indices, val_indices, test_indices)


def one_in_k_fold_split(data_list: Iterable,
                        n_splits: int,
                        split_index: int,
                        return_indices: bool,
                        random_state: int) -> Tuple[Iterable, Iterable, Iterable]:
    """Random split: One in k-fold split."""
    from sklearn import model_selection

    skf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_indices, test_indices = list(skf.split(data_list))[split_index]
    train_indices, val_indices = model_selection.train_test_split(train_indices,
                                                                  test_size=1 / (n_splits-1),
                                                                  random_state=random_state)

    if return_indices:
        return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()

    return _split_by_indices(data_list, train_indices, val_indices, test_indices)


def scaffold_split(smiles_list: Iterable,
                   return_indices: bool,
                   data_list: Iterable) -> Tuple[Iterable, Iterable, Iterable]:
    """Scaffold split: Deterministic scaffold split."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # Create mapping of the form {scaffold: [smiles_index_0, smiles_index_1, ...]}.
    scaffolds = defaultdict(list)
    for smiles_index, smiles in enumerate(smiles_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=True)
        scaffolds[scaffold].append(smiles_index)

    # Sort from the largest to smallest sets.
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: (len(x), x[0]), reverse=True)

    # Get train, validation, and test indices.
    train_indices, val_indices, test_indices = [], [], []

    train_cutoff = len(data_list) * 0.8
    val_cutoff = len(data_list) * (0.8+0.1)
    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_cutoff:
            train_indices.extend(scaffold_set)
        elif len(train_indices) + len(val_indices) + len(scaffold_set) <= val_cutoff:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    if return_indices:
        return train_indices, val_indices, test_indices

    return _split_by_indices(data_list, train_indices, val_indices, test_indices)


def randomized_scaffold_split(smiles_list: Iterable,
                              return_indices: bool,
                              data_list: Iterable,
                              random_state: int) -> Tuple[Iterable, Iterable, Iterable]:
    """Scaffold split: Randomized scaffold split."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # Create mapping of the form {scaffold: [smiles_index_0, smiles_index_1, ...]}.
    scaffolds = defaultdict(list)
    for smiles_index, smiles in enumerate(smiles_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=True)
        scaffolds[scaffold].append(smiles_index)

    # Randomly permutate scaffold sets.
    scaffold_sets = list(scaffolds.values())
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(scaffold_sets))
    scaffold_sets = [scaffold_sets[perm_index] for perm_index in perm]

    # Get train, validation, and test indices.
    train_indices, val_indices, test_indices = [], [], []

    train_cutoff = len(data_list) * 0.8
    val_cutoff = len(data_list) * (0.8 + 0.1)
    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_cutoff:
            train_indices.extend(scaffold_set)
        elif len(train_indices) + len(val_indices) + len(scaffold_set) <= val_cutoff:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    if return_indices:
        return train_indices, val_indices, test_indices

    return _split_by_indices(data_list, train_indices, val_indices, test_indices)


def _split_by_indices(data_list: Iterable,
                      train_indices: Iterable,
                      val_indices: Iterable,
                      test_indices: Iterable) -> Tuple[Iterable, Iterable, Iterable]:
    """Split `data_list` into train, validation, and test sets with provided indices."""
    if isinstance(data_list, list):
        return [data_list[index] for index in train_indices], \
               [data_list[index] for index in val_indices], \
               [data_list[index] for index in test_indices]
    elif isinstance(data_list, pd.DataFrame):
        return data_list.iloc[train_indices], \
               data_list.iloc[val_indices], \
               data_list.iloc[test_indices]
    else:
        from torch_geometric.data import InMemoryDataset

        if isinstance(data_list, InMemoryDataset):
            return data_list[train_indices], \
                   data_list[val_indices], \
                   data_list[test_indices]
        else:
            from dgl.data.utils import Subset
            from dgllife.data import MoleculeCSVDataset

            if isinstance(data_list, MoleculeCSVDataset):
                return Subset(data_list, train_indices), \
                       Subset(data_list, val_indices), \
                       Subset(data_list, test_indices)

    raise TypeError(f'Unsupported type of `data_list`: {type(data_list)}.')

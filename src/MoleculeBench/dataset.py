import os
import json
from argparse import Namespace

import pandas as pd
from rdkit import Chem


RAW_FILE = {
    'bace': 'bace.csv',
    'bbbp': 'BBBP.csv',
    'clintox': 'clintox.csv',
    'sider': 'sider.csv',
    'tox21': 'tox21.csv',
    'toxcast': 'toxcast_data.csv',
    'hiv': 'HIV.csv',
    'muv': 'muv.csv',
    'zinc2m': 'zinc_combined_apr_8_2019.csv'
}
code_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(code_dir, 'dataset')
info_path = os.path.join(code_dir, 'dataset_info.json')
with open(info_path, 'r') as fin:
    info = json.load(fin)


def filtered_and_canonicalized_dataset(dataset: str) -> pd.DataFrame:
    """Filter and canonicalize dataset, cache processed result."""
    filtered_path = os.path.join(dataset_dir, 'processed', f'{dataset}.csv')
    if not os.path.exists(filtered_path):
        print(f'Processing dataset "{dataset}"...')

        raw_path = os.path.join(dataset_dir, 'raw', RAW_FILE[dataset])
        os.makedirs(os.path.dirname(filtered_path), exist_ok=True)

        df = pd.read_csv(raw_path)
        smiles_column = dataset_info(dataset).smiles_column

        # Filter SMILES with "*", which causes size mismatch with pretrained embeddings.
        asterisk_sr = df.loc[:, smiles_column].apply(lambda smiles: "*" in smiles)
        df = df.loc[~asterisk_sr].reset_index(drop=True)

        # Filter unparsable SMILES.
        mol_sr = df.loc[:, smiles_column].apply(Chem.MolFromSmiles)
        df = df.loc[~mol_sr.isna()].reset_index(drop=True)

        # Canonicalize SMILES.
        canonicalizer = lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        df.loc[:, smiles_column] = df.loc[:, smiles_column].apply(canonicalizer)

        df.to_csv(filtered_path, index=False)

    return pd.read_csv(filtered_path)


def filtered_and_canonicalized_smiles(dataset: str) -> pd.Series:
    """Return filtered and canonicalized SMILES."""
    smiles_column = dataset_info(dataset).smiles_column
    smiles_sr = filtered_and_canonicalized_dataset(dataset).loc[:, smiles_column]
    return smiles_sr


def dataset_info(dataset: str) -> Namespace:
    assert dataset in info, f'Unsupported dataset: "{dataset}".'
    filtered_path = os.path.join(dataset_dir, 'processed', f'{dataset}.csv')
    return Namespace(**info[dataset], filtered_path=filtered_path)

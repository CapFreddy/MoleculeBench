# MoleculeBench

Benchmarking MoleculeNet datasets with unified pre-processing and data splits.

## Requirements

- Python 3.8.13
- pandas 1.3.4
- numpy 1.22.1
- torch_geometric 2.0.4
- dgl 0.7.2
- dgllife 0.2.8

`torch_geometric`, `dgl`, and `dgllife` are used for type checking and data splitting in `src/MoleculeBench/splitter.py`.

## Downloading Datasets

We provide the script for downloading the raw datasets from the official website of [MoleculeNet](https://moleculenet.org/).

```bash
cd src/MoleculeBench/datasets/raw
bash download.sh
```

For `zinc2m`, please [download](https://drive.google.com/file/d/1Y45zTHwWqtuliaTHRsLkr4RUtHW_jlLn/view?usp=sharing) manually and put the `.csv` file under `src/MoleculeBench/raw`.

## Installation

Install `MoleculeBench` to your local conda environment.

```bash
pip install -e .
```

## Usage
This package supports **loading** and **splitting** datasets.

### Loading
Load datasets or SMILES in datasets after three steps of pre-processing.

- Filter SMILES that contains asterisks (*), which causes size mismatch with pretrained embeddings
- Filter unparsable SMILES
- Canonicalize SMILES

```python
from MoleculeBench import filtered_and_canonicalized_dataset
from MoleculeBench import filtered_and_canonicalized_smiles

filtered_and_canonicalized_dataset('bace')
filtered_and_canonicalized_smiles('bace')
```

### Splitting
This package provides two APIs for splitting.

 * `train_val_test_split`: Standard random and scaffold splitting
 * `rigorous_train_val_test_split`: One in k-fold split & Randomized scaffold split

The APIs support returning split indices as well as directly splitting various data formats.

 * Regular python `list`
 * `numpy.ndarray`, `pandas.DataFrame`, `pandas.Series`
 * `torch_geometric.data.InMemoryDataset`
 * `dgllife.data.MoleculeCSVDataset`

```python
from MoleculeBench import train_val_test_split
from MoleculeBench import rigorous_train_val_test_split

train_val_test_split('bace')  # Scaffold split: Deterministic scaffold split.
train_val_test_split('clintox', random_state=42)  # Random split: Index after shuffling.

rigorous_train_val_test_split('bace', random_state=42)  # Scaffold split: Randomized scaffold split.
rigorous_train_val_test_split('clintox', n_splits=10, split_index=0, random_state=42)  # Random split: One in k-fold split.
```

### Dataset Information

Load dataset information as specified in `src/MoleculeBench/dataset_info.json`.

```python
from MoleculeBench import dataset_info

info = dataset_info('bace')
info.task_type
info.splitting
info.smiles_column
info.task_columns
```

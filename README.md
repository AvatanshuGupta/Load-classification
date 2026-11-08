# Load Classification

This project trains classifiers to identify the **behaviour** and **circuit**
of an electrical load from measured electrical features. It includes a
preparation notebook, a NumPy-based decision-tree/random-forest
implementation, and evaluation metrics for both classification tasks.

## Dataset

The source dataset is [`load_dataset_baseline_v3.xlsx`](load_dataset_baseline_v3.xlsx).
It contains 16,800 samples, with 22 numeric measurement features, a `behaviour`
label (`CZ`, `CP`, or `CC`), a `circuit` label (`RC`, `RL`, `LC`, `L`, `C`,
`R`, or `RLC`), and a `sample_id`.

The notebook [`data.ipynb`](data.ipynb) removes `sample_id`, label-encodes
both targets, and creates a 70/30 train/test split using `random_state=42`.
It saves the split datasets and label encoders in `artifacts/`.

## Requirements

- Python
- Packages listed in [`requirements.txt`](requirements.txt)
- Jupyter Notebook to run the preparation notebook (optional if the prepared
  files already exist in `artifacts/`)

Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

To prepare the training artifacts from the spreadsheet, launch Jupyter from
the repository root and run the notebook:

```bash
jupyter notebook data.ipynb
```

Run the notebook cells in order. The spreadsheet path is relative to the
repository root.

## Train and evaluate

From the repository root, run:

```bash
python train.py
```

The script loads the prepared files from `artifacts/`, trains separate custom
random forests for behaviour and circuit classification, evaluates each on
the test split, and writes model files and metrics to `results/`.

Default training options:

| Option | Default | Description |
|---|---:|---|
| `--n-trees` | `25` | Number of trees per forest |
| `--max-depth` | `20` | Maximum tree depth |
| `--min-samples-split` | `10` | Minimum samples required to split a node |
| `--n-feats` | `None` (square root of feature count) | Features considered at each split |
| `--seed` | `42` | Random seed |
| `--out-dir` | `results` | Directory for models and metrics |

For example:

```bash
python train.py --n-trees 10 --max-depth 12 --seed 7 --out-dir results
```

The training artifacts must exist before running `train.py`; to create them,
run the notebook first. Run commands from the repository root so the relative
input paths resolve correctly.

## Outputs

The default run creates:

- `results/rf_behaviour-t25-d20.pkl` — behaviour classifier
- `results/rf_circuit-t25-d20.pkl` — circuit classifier
- `results/performance.json` — configuration, accuracy, macro F1, confusion
  matrices, class labels, and training times

The metric helpers in [`score.py`](score.py) handle feature-matrix conversion,
accuracy, macro F1, and confusion matrices. The tree and forest
implementations are in [`model.py`](model.py).

## Recorded results

The checked-in [`results/performance.json`](results/performance.json) records
the following test-set scores for its configuration (25 trees, maximum depth
20, minimum split size 10, seed 42):

| Target | Accuracy | Macro F1 |
|---|---:|---:|
| Behaviour | 0.9079 | 0.9084 |
| Circuit | 0.8530 | 0.8358 |

Training times depend on the machine and may differ from the times in the
results file.

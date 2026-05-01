# PonziSense

PonziSense is an explainable detector for Ponzi smart contracts. This repository contains the paper-reproduction engineering code only: model definitions, preprocessing, training/evaluation entry points, graph/rationale utilities, baseline scripts, and RQ experiment scripts.

## What is included

- Core pipeline: `preprocess_dataset.py`, `train.py`, `evaluate.py`, `predict.py`, `inference.py`
- Model and method code: `models/`, `data/`, `graph/`, `utils/`, `configs/`
- Paper experiment scripts: `experiments/`, `baseline/`, `hpo/`
- Solidity parser source: `parser/tree-sitter-solidity/` plus `parser/build.py`
- Compact release dataset: `datafiles/ponzi_e_release.csv`

## What is intentionally excluded

- Raw Solidity source-code datasets and experimental data dumps
- Generated train/validation/test splits under `datafiles/processed/`
- Model checkpoints, `.bin`, `.pt`, `.pth`, `.ckpt`, `.safetensors`, and generated parser binaries
- Web/demo/system prototype code and other non-paper software engineering artifacts
- Paper LaTeX source, build outputs, logs, caches, virtual environments, and notebooks

## Dataset

`datafiles/ponzi_e_release.csv` is a compact, source-free view of Ponzi-E:

- Total rows: 8,233
- Ponzi rows: 744 (`label=1`)
- non-Ponzi rows: 7,489 (`label=0`)
- Split: 60/20/20 train/validation/test

The Solidity `code` column is removed for release size and safety. When an address is available, it is stored in `address`; otherwise, `code_hash` is kept as an irreversible identifier.

To materialize a training CSV for address-backed rows:

```bash
python scripts/materialize_dataset_from_addresses.py   --input datafiles/ponzi_e_release.csv   --output datafiles/PonziDataset_20221114_explain_augmented_negatives.csv
```

Sourcify is used first. If you set `ETHERSCAN_API_KEY`, the script also tries Etherscan as a fallback.

## Reproduce the pipeline

```bash
pip install -r requirements.txt
cd parser && python build.py && cd ..
python preprocess_dataset.py
python train.py
python evaluate.py
```

`preprocess_dataset.py` expects a full `code,label,explain` CSV at `datafiles/PonziDataset_20221114_explain_augmented_negatives.csv`. The compact release CSV does not contain source code by design.

## Label convention

PonziSense follows the paper convention: `label=1` means Ponzi and `label=0` means non-Ponzi.

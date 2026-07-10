# QPnP-MMFND: Quantum-inspired Plug-and-Play Framework for Multimodal Multi-domain Fake News Detection

QPnP-MMFND is a quantum-inspired plug-and-play framework for multimodal multi-domain fake news detection. The core method in the paper contains two complementary modules:

- Quantum Ambiguity Modeling (QAM): encodes textual and visual features as quantum states to capture cross-domain uncertainty and multimodal ambiguity.
- Quantum-inspired Interference Fusion (QIF): uses interference-inspired formulations and phase differences to model fine-grained cross-modal interactions.

This repository provides the implementation, preprocessing scripts, and training entry points for the paper.

## Requirements

Install the environment with:

```bash
pip install -r requirements.txt
```

The project uses PyTorch, Transformers, Chinese-CLIP, and pretrained Roberta/MAE checkpoints.

## Directory Structure

```text
├── CNN_architectures/
├── model/
├── pretrained_model/
├── util/
├── utils/
├── data/
│   ├── train_loader.pkl
│   ├── train_clip_loader.pkl
│   ├── train_origin.csv
│   └── ...
├── Weibo_21/
│   ├── train_loader.pkl
│   ├── train_clip_loader.pkl
│   ├── train_datasets.xlsx
│   └── ...
├── clip_data_pre.py
├── data_pre.py
├── main.py
├── models_mae.py
├── run.py
├── weibo21_generate_pkl.py
├── weibo21_generate_clip_pkl.py
└── requirements.txt
```

## Data Preparation

### Datasets

- Weibo data should be placed in `./data`.
- Weibo21 data should be placed in `./Weibo_21`.

The current code expects the following split files:

- Weibo: `train_origin.csv`, `val_origin.csv`, `test_origin.csv`
- Weibo21: `train_datasets.xlsx`, `val_datasets.xlsx`, `test_datasets.xlsx`

### Preprocessing

Use the preprocessing scripts to generate cached loader files before training:

- `data_pre.py`
- `clip_data_pre.py`
- `weibo21_generate_pkl.py`
- `weibo21_generate_clip_pkl.py`

These scripts create `*_loader.pkl` and `*_clip_loader.pkl` files to speed up data loading.

### Dataset Notes

- Weibo21 follows the dataset setting used by Ying et al. (2023).
- Weibo follows the dataset setting used by Tong et al. (2024), with domain labels already included in the processed data.

If you use the released processed Weibo data, you can skip the raw data preparation stage.

## Pretrained Models

Place the pretrained resources in the paths expected by the code:

- Roberta / Chinese-Roberta: `./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/`
- Roberta vocabulary: `./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt`
- MAE: place the pretrained MAE files in the project root or the path referenced by the model code
- Chinese-CLIP: place the pretrained CLIP files in the project root or the path referenced by the model code

## Training

The main training entry is `main.py`.

### Weibo

```bash
python main.py --dataset weibo --root_path ./data/
```

### Weibo21

```bash
python main.py --dataset weibo21 --root_path ./Weibo_21/
```

### Common Arguments

- `--model_name`: model name, default is `QMMFND`
- `--dataset`: `weibo` or `weibo21`
- `--epoch`: training epochs, default `50`
- `--batchsize`: batch size, default `64`
- `--gpu`: GPU id, default `1`
- `--early_stop`: early stopping patience, default `5`
- `--max_len`: maximum token length, default `197`

Example:

```bash
python main.py --dataset weibo --gpu 0 --batchsize 32 --epoch 50
```

## Model Notes

The `model/` directory contains the QAM/QIF variants used in the paper, including:

- `MMDFND+QAM.py`
- `MMDFND+QIF.py`
- `m3FEND+QAM.py`
- `m3FEND+QIF.py`
- `mdfend+QAM.py`
- `mdfend+QIF.py`
- `eddfn+QAM.py`
- `eddfn+QIF.py`
- `DAMMDFND+QAM.py`
- `DAMMDFND+QIF.py`

The current training pipeline loads the trainer according to the `model_name` argument in `run.py`.

## Citation

If you use this repository in your research, please cite the corresponding QPnP-MMFND paper.



# QMMFND

This is an official implementation for QMMFND (Quantum-inspired Multi-Modal Multi-Domain Fake News Detection). The project integrates quantum computing principles with multi-modal deep learning techniques to accurately detect fake news from multiple domains. If you like this repo, don't forget to give a star!
## Requirements
You can run `pip install -r requirements.txt` to deploy the environment.

## Directory Structure

```
QMMFND/
├── CNN_architectures/          # CNN model architectures
│   ├── fp16_util.py
│   ├── lenet5_pytorch.py
│   ├── pytorch_resnet.py
│   ├── pytorch_vgg_implementation.py
│   ├── pytorch_efficientnet.py
│   ├── pytorch_inceptionet.py
│   ├── nn.py
│   └── unet.py
├── model/                       # Core model implementations
│   ├── QMMFND.py               # Main model
│   ├── bert.py                 # BERT-related modules
│   ├── entangle.py             # Quantum entanglement alignment
│   ├── pivot.py                # Keypoint modules
│   ├── layers.py               # Custom layers
│   ├── wavefunction_quantum.py  # Wave function quantum encoder
│   ├── clip_quantum_encoder.py  # CLIP quantum fusion
│   └── bert_mae_mixed_state_encoder.py  # BERT/MAE mixed-state encoder
├── util/                        # Utility functions
│   ├── crop.py
│   ├── datasets.py
│   ├── lars.py
│   ├── lr_decay.py
│   ├── lr_sched.py
│   ├── misc.py
│   └── pos_embed.py
├── utils/                       # Data loading and utilities
│   ├── utils.py
│   ├── dataloader.py
│   ├── clip_dataloader.py
│   ├── weibo21_clip_dataloader.py
│   └── finefake_dataloader.py
├── data/                        # Weibo dataset (requires manual download)
│   ├── train_clip_loader.pkl
│   ├── train_loader.pkl
│   └── train_origin.csv
├── Weibo_21/                    # Weibo21 dataset (requires application)
│   ├── train_clip_loader.pkl
│   ├── train_loader.pkl
│   └── train_datasets.xlsx
├── pretrained_model/            # Pre-trained models directory
│   └── chinese_roberta_wwm_base_ext_pytorch/
├── main.py                      # Training entry point
├── run.py                       # Run script
├── models_mae.py                # MAE model implementation
├── data_pre.py                  # Weibo data preprocessing
├── clip_data_pre.py             # Weibo CLIP data preprocessing
├── weibo21_data_pre.py          # Weibo21 data preprocessing
├── weibo21_clip_data_pre.py     # Weibo21 CLIP data preprocessing
├── requirements.txt             # Dependencies list
└── README.md                    # Documentation
```

## Data Preparation

### 1. Data Acquisition

#### Weibo Dataset
For the Weibo dataset, we follow the work from[(Tong et al.， 2024)]( https://github.com/yutchina/MMDFND)


#### Weibo21 Dataset
For the Weibo21 dataset, we follow the work from [(Ying et al.， 2023)](https://github.com/yingqichao/fnd-bootstrap). You should send an email to Dr. [Qiong Nan](mailto:nanqiong19z@ict.ac.cn) to get the complete multimodal multi-domain dataset Weibo21.

### 2. Data Storage

- Place the processed Weibo data in the `./data` directory
- Place the Weibo21 data in the `./Weibo_21` directory

### 3. Data Preprocessing

Execute data preprocessing before training to accelerate data loading:
```bash
# Weibo data preprocessing
python data_pre.py
python clip_data_pre.py

# Weibo21 data preprocessing
python weibo21_generate_pkl.py
python weibo21_generate_clip_pkl.py
```

## Pre-trained Models

### 1. RoBERTa Chinese Pre-trained Model

Download the pretrained Roberta model from [Roberta](https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing) and move all files into the `./pretrained_model` directory.

### 2. MAE Pre-trained Model
Download the pretrained MAE model from ["Masked Autoencoders： A PyTorch Implementation"](https://github.com/facebookresearch/mae) and move all files into the root directory.
### 3. CLIP Chinese Pre-trained Model
Download the pretrained CLIP model from ["Chinese-CLIP"](https://github.com/OFA-Sys/Chinese-CLIP) and move all files into the root directory.

## Training

### Basic Training Command

```bash
python main.py
```

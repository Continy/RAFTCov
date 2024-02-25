# RAFTCov

RAFTCov provides a simple way to use RAFT structure to estimate covariance matrix of optical flow. The core part which calculats the memory cost is replaced by a simple attention module.

## Installation

### Docker

Docker Hub:

```bash
docker zihaozhang/flowformer:v1.1
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA 12.1
- cv2, cupy
- numpy, scipy (cov_toolkit), matplotlib, pyyaml
- wandb (optional)

## Usage

### Training in TartanAir dataset

Please check DataLoader in
[core/datasets.py](core/datasets.py)

```bash
python train.py --config your_config.yaml
```

### Evaluation

- Save your model in `./models/`

- Drag all files in [./cov_toolkit/](./cov_toolkit/) to the root of the dataset, and run:

```bash
python multi_viz.py
python plot_and_correlation.py
python variance_mse_correlation_analysis.py
```

- Then the result including the covariance matrix, the flow and the image will be saved in `./result/`

- Check common_sparsification_plot.png and curves.png for the result

## TODO

- [ ] Add more datasets
- [ ] YAML support for evaluation
- [x] FlowNet support
  - [x] Training script
  - [x] Evaluation script
  - [x] Pretrained model
- [ ] StereoNet support
  - [x] Training script
  - [ ] Evaluation script
  - [ ] Pretrained model

## Others

This repo gives a small tool to estimate covariance matrix of optical flow.
<https://github.com/haleqiu/cov_toolkit>

For the original version of this work, please see: <https://github.com/NVlabs/PWC-Net>

Thanks to this repo for a higher pytorch version and a easier way to use:
<https://github.com/sniklaus/pytorch-pwc>

Decoder part is from:
<https://github.com/drinkingcoder/FlowFormer-Official>

FasterViT:
<https://github.com/NVlabs/FasterViT>

FlowFormer-like covaraince network:
<https://github.com/Continy/FlowFormer>

Flow & Stereo backbone:
<https://github.com/castacks/tartanvo>

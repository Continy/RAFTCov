import numpy as np
import torch.nn.functional as F
from yacs.config import CfgNode as CN
import yaml


def preprocess(img):
    #input: torch.Tensor, shape: (B, C, H, W)

    img = img / 255.0

    B, C, H, W = img.shape
    W_ = int(np.floor(np.ceil(W / 64.0) * 64.0))
    H_ = int(np.floor(np.ceil(H / 64.0) * 64.0))

    img = F.interpolate(input=img,
                        size=(H_, W_),
                        mode='bilinear',
                        align_corners=False)

    return img, W, H, W_, H_


def reverse(img, W, H, W_, H_, is_flow=False):
    img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
    if is_flow:
        img[:, 0, :, :] *= float(W) / float(W_)
        img[:, 1, :, :] *= float(H) / float(H_)
    return img


def build_cfg(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = CN(yaml.safe_load(f))
    return cfg

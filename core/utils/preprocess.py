import torch
import sys
import numpy as np
import PIL


def preprocess(img):
    #input: torch.Tensor, shape: (B, C, H, W)

    img = img / 255.0

    B, C, H, W = img.shape
    W_ = int(np.floor(np.ceil(W / 64.0) * 64.0))
    H_ = int(np.floor(np.ceil(H / 64.0) * 64.0))

    img = torch.nn.functional.interpolate(input=img,
                                          size=(H_, W_),
                                          mode='bilinear',
                                          align_corners=False)

    return img, W, H, W_, H_

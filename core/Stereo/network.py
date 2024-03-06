import sys
import torch
import torch.nn as nn
from .gru import GaussianGRU, HourglassDecoder
from .StereoNet import StereoNet7
import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as v2
from ..utils import Utility as Utility


class StereoFeature(nn.Module):

    def __init__(self):
        super(StereoFeature, self).__init__()
        self.stereo = StereoNet7()

    def forward(self, tenOne, tenTwo):
        _, cropsize = Utility.getCropMargin(tenOne.shape)
        transform = v2.Compose([
            v2.CenterCrop(cropsize),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225]),
            v2.ToTensor()
        ])
        tenOne, tenTwo = transform(tenOne).cuda(), transform(tenTwo).cuda()
        stereo, context, memory, costmap, cats = self.stereo(
            torch.cat((tenOne, tenTwo), dim=1))
        return stereo, context, memory, costmap, cats


class RAFTCovWithStereoNet7(nn.Module):

    def __init__(self, cfg):
        super(RAFTCovWithStereoNet7, self).__init__()

        self.feature = StereoFeature()
        #self.netGaussian = GaussianGRU(cfg) # AttentionDecoder
        self.decoder = HourglassDecoder()

    def forward(self, tenOne, tenTwo):
        stereo, context, memory, costmap, cats = self.feature(tenOne, tenTwo)
        cov_preds = self.decoder(context, cats)
        return stereo, cov_preds

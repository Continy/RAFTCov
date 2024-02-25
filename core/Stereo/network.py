import sys
import torch
import torch.nn as nn
from .gru import GaussianGRU
from .StereoNet import StereoNet7


class StereoFeature(nn.Module):

    def __init__(self):
        super(StereoFeature, self).__init__()
        self.stereo = StereoNet7()

    def forward(self, tenOne, tenTwo):
        flow, context, memory, costmap = self.stereo(tenOne, tenTwo)
        return flow, context, memory, costmap


class RAFTCovWithStereoNet7(nn.Module):

    def __init__(self, cfg):
        super(RAFTCovWithStereoNet7, self).__init__()

        self.feature = StereoFeature()
        self.netGaussian = GaussianGRU(cfg)

    def forward(self, tenOne, tenTwo):
        flow, context, memory, costmap = self.feature(tenOne, tenTwo)
        cov_preds = self.netGaussian(context, memory, costmap)
        return flow, cov_preds

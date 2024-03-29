import sys
import torch
import torch.nn as nn
from .gru import GaussianGRU
from .PWCFlowNet import PWCDCNet, PWCFeature
from .PWCCovNet import CovFeature


class RAFTCovWithPWCNet(nn.Module):

    def __init__(self, cfg):
        super(RAFTCovWithPWCNet, self).__init__()

        self.feature = PWCFeature()
        self.netGaussian = GaussianGRU(cfg)
        if cfg.tartanvo_model is not None:
            self.feature.load_tartanvo_weight(cfg.tartanvo_model)
            #self.feature.init_cnet()

    def forward(self, tenOne, tenTwo):
        flow, context, memory, costmap = self.feature(tenOne, tenTwo)
        cov_preds = self.netGaussian(context, memory, costmap)
        return flow, cov_preds


class MonoCovWithFasterViT(nn.Module):

    def __init__(self, cfg):
        super(MonoCovWithFasterViT, self).__init__()

        self.feature = PWCFeature()
        self.netGaussian = GaussianGRU(cfg)
        if cfg.tartanvo_model is not None:
            self.feature.load_tartanvo_weight(cfg.tartanvo_model)

    def forward(self, tenOne, tenTwo):
        flow, mem = self.feature(tenOne, tenTwo)
        cov_preds = self.netGaussian(mem)
        return flow, cov_preds


class PWCCov(nn.Module):

    def __init__(self, cfg):
        super(PWCCov, self).__init__()

        self.feature = PWCFeature()
        self.netGaussian = CovFeature()
        if cfg.tartanvo_model is not None:
            self.feature.load_tartanvo_weight(cfg.tartanvo_model)
            self.netGaussian.load_tartanvo_weight(cfg.tartanvo_model)
            #self.feature.init_cnet()

    def forward(self, tenOne, tenTwo):
        flow, _, _, _ = self.feature(tenOne, tenTwo)
        cov, _, _, _ = self.netGaussian(tenOne, tenTwo)
        return flow, cov

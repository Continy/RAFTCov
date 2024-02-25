import sys
import torch
import torch.nn as nn
from .gru import GaussianGRU
from .StereoNet import StereoNet7
from ..utils import Utility as Utility


class StereoFeature(nn.Module):

    def __init__(self):
        super(StereoFeature, self).__init__()
        self.stereo = StereoNet7()
        self.transform = Utility.Compose([
            Utility.DownscaleFlow(scale=4.),
            Utility.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225],
                              rgbbgr=False,
                              keep_old=True),
            Utility.ToTensor()
        ])

    def forward(self, tenOne, tenTwo):
        _, cropsize = Utility.getCropMargin(tenOne.shape)
        cropcenter_transform = Utility.CropCenter(cropsize,
                                                  fix_ratio=False,
                                                  scale_w=1.0,
                                                  scale_disp=False)
        imgs = Utility.frame2Sample(tenOne, tenTwo)
        imgs = self.transform(cropcenter_transform(imgs))

        img1, img2 = img1['img0_norm'].cuda(), img2['img1_norm'].cuda()
        flow, context, memory, costmap = self.stereo(
            torch.cat((tenOne, tenTwo), dim=1))
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

import sys
import torch
from .encoder import Encoder
from .decoder import Decoder, AttentionLayer
from .refiner import Refiner, GaussianGRU


class Network(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.netExtractor = Encoder()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        self.netGaussian = GaussianGRU(cfg)

    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)
        context = torch.cat([tenOne[-5], tenTwo[-5]], dim=1)
        #memory = objEstimate
        cost_map = objEstimate['tenFlow']
        cov_preds = self.netGaussian(context, objEstimate['tenFeat'], cost_map)
        return (objEstimate['tenFlow'] +
                self.netRefiner(objEstimate['tenFeat'])) * 20.0, cov_preds

    # end


# end

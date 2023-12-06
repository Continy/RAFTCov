import sys
import torch
from encoder import Encoder, SelfAttentionLayer
from decoder import Decoder
from refiner import Refiner, GaussianEncoder


class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.netExtractor = Encoder()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)
        print(objEstimate['tenFlow'].shape)
        print(objEstimate['tenFeat'].shape)
        return (objEstimate['tenFlow'] +
                self.netRefiner(objEstimate['tenFeat'])) * 20.0

    # end


# end

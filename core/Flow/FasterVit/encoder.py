import torch
import torch.nn as nn
import torch.nn.functional as F
from ...fastervit.faster_vit_any_res import FasterViTLayer


class MemoryEncoder(nn.Module):

    def __init__(self):
        super(MemoryEncoder, self).__init__()

        self.encoder = FasterViTLayer(dim=128,
                                      depth=6,
                                      num_heads=8,
                                      ct_size=2,
                                      window_size=8,
                                      mlp_ratio=4,
                                      drop=0.1,
                                      downsample=True,
                                      input_resolution=(128, 160))

    def forward(self, memory):
        return self.encoder(memory)


class ContextEncoder(nn.Module):

    def __init__(self):
        super(ContextEncoder, self).__init__()

        self.encoder = FasterViTLayer(dim=256,
                                      depth=4,
                                      num_heads=4,
                                      ct_size=2,
                                      window_size=7,
                                      mlp_ratio=4,
                                      drop=0.1,
                                      downsample=False,
                                      input_resolution=(64, 80))

    def forward(self, context):
        return self.encoder(context)


class CostMapEncoder(nn.Module):

    def __init__(self):
        super(CostMapEncoder, self).__init__()

        self.encoder = FasterViTLayer(dim=256,
                                      depth=5,
                                      num_heads=8,
                                      ct_size=2,
                                      window_size=8,
                                      mlp_ratio=4,
                                      drop=0.1,
                                      downsample=False,
                                      input_resolution=(64, 80))

    def forward(self, cost_map):
        return self.encoder(cost_map)


if __name__ == '__main__':
    input = torch.rand(1, 128, 64, 80)
    model = ContextEncoder()
    out = model(input)
    print(out.shape)

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .correlation import correlation  # the custom cost volume layer


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000
                          **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(
            tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class AttentionLayer(nn.Module):

    def __init__(self, cfg, dropout=0.):
        super(AttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.pos = PositionalEncoding2D(cfg.dim)
        self.q, self.k, self.v = nn.Linear(cfg.dim, cfg.dim), nn.Linear(
            512, cfg.dim), nn.Linear(512, cfg.dim)
        self.att = nn.MultiheadAttention(cfg.dim,
                                         cfg.num_heads,
                                         dropout=cfg.dropout,
                                         batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(),
                                 nn.Dropout(cfg.dropout),
                                 nn.Linear(cfg.dim, cfg.dim),
                                 nn.Dropout(cfg.dropout))

    def forward(self, query, key, value, memory, cov):
        if key is None and value is None:
            key, value = self.k(memory), self.v(memory)
            B, C, H, W = key.shape
            key = key.reshape(B, H * W * C // self.dim, self.dim)
            value = value.reshape(B, H * W * C // self.dim, self.dim)
        B, C, H, W = query.shape
        query = query.reshape(B, H * W * C // self.dim, self.dim)
        shortcut = query
        query = self.norm1(query)

        cov_pos_embed = self.pos(cov.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        cov = torch.cat([cov, cov_pos_embed], dim=1)
        B, C, H, W = cov.shape
        cov = cov.reshape(B, H * W * C // self.dim, self.dim)

        q = self.q(torch.cat([query, cov], dim=1))
        k, v = key, value
        x = self.att(q, k, v)[0]

        x = self.proj(torch.cat([x, shortcut], dim=1))
        x = x + self.ffn(self.norm2(x))
        C = 4 * self.cfg.mixtures + 4
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x, k, v


class Decoder(torch.nn.Module):

    def __init__(self, intLevel):
        super().__init__()

        self.backwarp_tenGrid = {}
        self.backwarp_tenPartial = {}

        intPrevious = [
            None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2,
            81 + 128 + 2 + 2, 81, None
        ][intLevel + 1]
        intCurrent = [
            None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2,
            81 + 128 + 2 + 2, 81, None
        ][intLevel + 0]

        if intLevel < 6:
            self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2,
                                                      out_channels=2,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1)
        if intLevel < 6:
            self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious +
                                                      128 + 128 + 96 + 64 + 32,
                                                      out_channels=2,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1)
        if intLevel < 6:
            self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625,
                                None][intLevel + 1]

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128,
                            out_channels=96,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                            out_channels=2,
                            kernel_size=3,
                            stride=1,
                            padding=1))

    # end

    def forward(self, tenOne, tenTwo, objPrevious):
        tenFlow = None
        tenFeat = None

        if objPrevious is None:
            tenFlow = None
            tenFeat = None

            tenVolume = torch.nn.functional.leaky_relu(
                input=correlation.FunctionCorrelation(tenOne=tenOne,
                                                      tenTwo=tenTwo),
                negative_slope=0.1,
                inplace=False)

            tenFeat = torch.cat([tenVolume], 1)

        elif objPrevious is not None:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])

            tenVolume = torch.nn.functional.leaky_relu(
                input=correlation.FunctionCorrelation(
                    tenOne=tenOne,
                    tenTwo=self.backwarp(tenInput=tenTwo,
                                         tenFlow=tenFlow * self.fltBackwarp)),
                negative_slope=0.1,
                inplace=False)

            tenFeat = torch.cat([tenVolume, tenOne, tenFlow, tenFeat], 1)

        # end

        tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

        tenFlow = self.netSix(tenFeat)

        return {'tenFlow': tenFlow, 'tenFeat': tenFeat}

    # end
    def backwarp(self, tenInput, tenFlow):
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(
                1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
            tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(
                1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat(
                [tenHor, tenVer], 1).cuda()
        # end

        if str(tenFlow.shape) not in self.backwarp_tenPartial:
            self.backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones(
                [tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])
        # end

        tenFlow = torch.cat([
            tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)),
            tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))
        ], 1)
        tenInput = torch.cat(
            [tenInput, self.backwarp_tenPartial[str(tenFlow.shape)]], 1)

        tenOutput = torch.nn.functional.grid_sample(
            input=tenInput,
            grid=(self.backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(
                0, 2, 3, 1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)

        tenMask = tenOutput[:, -1:, :, :]
        tenMask[tenMask > 0.999] = 1.0
        tenMask[tenMask < 1.0] = 0.0

        return tenOutput[:, :-1, :, :] * tenMask


# end

if __name__ == '__main__':
    netsix = Decoder(6).cuda()
    input = torch.randn(1, 196, 16, 7).cuda()
    output = netsix(input, input, None)
    print('shape of output:')
    print(output['tenFlow'].shape)
    print(output['tenFeat'].shape)

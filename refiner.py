import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import GaussianEncoder
from decoder import Decoder


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init


class GaussianGRU(nn.Module):

    def __init__(self, cfg):
        super(GaussianGRU, self).__init__()
        self.cfg = cfg
        self.iters = cfg.gru_iters
        self.proj = nn.Conv2d(565, 512, 1)  # TODO: 128 or 256
        self.gaussian = GaussianUpdateBlock(cfg, hidden_dim=128)

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)

        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8 * H, 8 * W)

    def forward(self, context):
        covs0, covs1 = initialize_flow(context)
        covs0 = covs0.repeat(1, self.cfg.mixtures, 1, 1)
        covs1 = covs1.repeat(1, self.cfg.mixtures, 1, 1)
        context = self.proj(context)
        net, inp = torch.split(context, [256, 256], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        cov = covs1 - covs0
        corr = []  #TODO
        for i in range(self.iters):
            net, delta_covs, up_mask, inp = self.gaussian(net, inp, corr, cov)
            covs1 = covs0 + delta_covs
        cov_up = self.upsample_flow(covs1 - covs0, up_mask)
        return cov_up


class GaussianHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, mixtures=9):
        super(GaussianHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2 * mixtures, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim,
                                hidden_dim, (1, 5),
                                padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim,
                                hidden_dim, (1, 5),
                                padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim,
                                hidden_dim, (1, 5),
                                padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim,
                                hidden_dim, (5, 1),
                                padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim,
                                hidden_dim, (5, 1),
                                padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim,
                                hidden_dim, (5, 1),
                                padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class GaussianUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = GaussianEncoder()
        self.gaussian = SepConvGRU(hidden_dim=hidden_dim,
                                   input_dim=128 + hidden_dim + hidden_dim)
        self.gaussian_head = GaussianHead(hidden_dim,
                                          hidden_dim=256,
                                          mixtures=args.mixtures)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, cov):
        features = self.encoder(cov, corr)
        inp = torch.cat([inp, features], dim=1)
        net = self.gaussian(net)
        delta_covs = self.gaussian_head(net)
        up_mask = .25 * self.mask(net)
        return net, delta_covs, up_mask, inp


class Refiner(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 +
                            32,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=2,
                            dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=4,
                            dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=96,
                            kernel_size=3,
                            stride=1,
                            padding=8,
                            dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=16,
                            dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=2,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            dilation=1))

    # end

    def forward(self, tenInput):
        return self.netMain(tenInput)

    # end


if __name__ == '__main__':
    input = torch.randn(1, 565, 112, 256)
    refiner = Refiner()
    output = refiner(input)
    print(output.shape)

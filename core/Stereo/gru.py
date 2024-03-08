import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionLayer


def distance_grid(batch, ht, wd):
    center_x, center_y = wd / 2, ht / 2
    x_coords = torch.arange(wd).float() - center_x
    y_coords = torch.arange(ht).float() - center_y
    distance = torch.sqrt(x_coords[None, :]**2 + y_coords[:, None]**2)
    distance = distance[None, None, ...].repeat(batch, 1, 1, 1)

    return distance.to(torch.float32)


def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = distance_grid(N, H, W).to(img.device)
    mean_init = distance_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init


class GaussianGRU(nn.Module):

    def __init__(self, cfg):
        super(GaussianGRU, self).__init__()
        self.cfg = cfg
        self.iters = cfg.gru_iters
        #downsample x2
        self.proj = nn.Sequential(
            nn.Conv2d(448, cfg.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mem_proj = nn.Conv2d(192, 64, 1, padding=0)
        self.costmap_proj = nn.Conv2d(192, 4, 1, padding=0)
        self.att = AttentionLayer(cfg)
        self.gaussian = GaussianUpdateBlock(cfg, hidden_dim=cfg.dim)

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

    def forward(self, context, memory, cost_map):

        cov_preds = []
        memory = self.mem_proj(memory)
        memory = memory.permute(0, 2, 3, 1)
        covs0, covs1 = initialize_flow(context)
        covs0 = covs0.repeat(1, self.cfg.mixtures, 1, 1)
        covs1 = covs1.repeat(1, self.cfg.mixtures, 1, 1)
        context = self.proj(context)
        cost_map = self.costmap_proj(cost_map)
        net, inp = torch.split(context, [self.cfg.dim, self.cfg.dim], dim=1)
        net = torch.tanh(net)
        inp = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)(inp)

        key, value = None, None
        up_mask = None

        for i in range(self.iters):
            cost, key, value = self.att(cost_map, key, value, memory, covs1)
            corr = torch.cat([cost, cost_map], dim=1)  #C:4*mixtures+6
            cov = covs1 - covs0
            net, delta_covs, up_mask = self.gaussian(net, inp, corr, cov)
            covs1 = covs0 + delta_covs
            cov_up = self.upsample_flow(covs1 - covs0, up_mask)
            cov_preds.append(cov_up)
        return cov_preds


class GaussianHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, mixtures=9):
        super(GaussianHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, mixtures, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))

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


class GaussianEncoder(nn.Module):

    def __init__(self, cfg):
        super(GaussianEncoder, self).__init__()

        self.convc1 = nn.Conv2d(2 * cfg.mixtures + 12, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(cfg.mixtures, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, cov, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(cov))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)
        return torch.cat([out, cov], dim=1)  # 126+2*mixtures


class GaussianUpdateBlock(nn.Module):

    def __init__(self, cfg, hidden_dim=128):
        super().__init__()
        self.cfg = cfg
        self.encoder = GaussianEncoder(cfg)
        self.gaussian = SepConvGRU(hidden_dim=hidden_dim,
                                   input_dim=126 + 2 * cfg.dim + cfg.mixtures)
        self.gaussian_head = GaussianHead(hidden_dim,
                                          hidden_dim=hidden_dim,
                                          mixtures=cfg.mixtures)
        self.mask = nn.Sequential(nn.Conv2d(cfg.dim, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, cov):
        features = self.encoder(cov, corr)
        inp = torch.cat([inp, features], dim=1)  #C:126+2*mixtures+dim

        net = self.gaussian(net, inp)  #(B,128,H,W)
        delta_covs = self.gaussian_head(net)
        up_mask = .25 * self.mask(net)
        return net, delta_covs, up_mask


from .PSM import Hourglass


class HourglassDecoder(nn.Module):

    def __init__(self, act_fun='selu'):
        super(HourglassDecoder, self).__init__()
        if act_fun == 'relu':
            self.actfun = F.relu
        elif act_fun == 'selu':
            self.actfun = F.selu
        else:
            print('Unknown activate function', act_fun)
            self.actfun = F.relu
        self.deconv_c7 = nn.ConvTranspose2d(896,
                                            320,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1)  # 1/16
        self.deconv_c8 = nn.ConvTranspose2d(576,
                                            192,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1)  # 1/8
        self.conv_c8 = Hourglass(2, 192, 0)  # 1/8
        self.deconv_c9 = nn.ConvTranspose2d(384,
                                            128,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1)  # 1/4
        self.conv_c9 = Hourglass(2, 128, 0)  # 1/4
        self.deconv_c10 = nn.ConvTranspose2d(256,
                                             64,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1)  # 1/2
        self.conv_c10 = Hourglass(2, 64, 0)  # 1/2
        self.deconv_c11 = nn.ConvTranspose2d(128,
                                             64,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1)  # 1/1
        self.conv_c12 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.conv_c13 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        self.deconv_c7_2 = nn.ConvTranspose2d(512,
                                              512,
                                              kernel_size=4,
                                              stride=2,
                                              padding=1)  # 1/32

    def forward(self, x, cats):
        cat0, cat1, cat2, cat3, cat4 = cats

        x = self.deconv_c7_2(x)  # 1/32 - 512
        x = nn.LeakyReLU(inplace=True, negative_slope=0.1)(x)
        x = torch.cat((x, cat4), dim=1)  # - 896
        x = self.deconv_c7(x)  # 1/16 - 320
        x = nn.LeakyReLU(inplace=True, negative_slope=0.1)(x)
        x = torch.cat((x, cat3), dim=1)  # - 576 28,40
        x = self.deconv_c8(x)  # 1/8 - 192
        x = nn.LeakyReLU(inplace=True, negative_slope=0.1)(x)
        x = self.conv_c8(x)
        x = torch.cat((x, cat2), dim=1)  # - 384 56,80
        x = self.deconv_c9(x)  # 1/4 - 128
        x = nn.LeakyReLU(inplace=True, negative_slope=0.1)(x)
        x = self.conv_c9(x)
        x = torch.cat((x, cat1), dim=1)  # - 256 112,160
        x = self.deconv_c10(x)  # 1/2 - 64
        x = self.conv_c10(x)
        x = torch.cat((x, cat0), dim=1)  # - 128 224,320
        x = self.deconv_c11(x)  # 1/1 - 64
        x = self.conv_c12(x)

        out0 = self.conv_c13(x)
        return out0

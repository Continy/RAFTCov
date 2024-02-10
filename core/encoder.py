import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianEncoder(nn.Module):

    def __init__(self, cfg):
        super(GaussianEncoder, self).__init__()

        self.convc1 = nn.Conv2d(1156, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
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


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)
        if not stride == 1:
            self.norm3 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CorrBlock:

    def __init__(self, fmap1, fmap2, num_levels=4, radius=8):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.c3 = ResidualBlock(16, 32, stride=2).cuda()
        self.c2 = ResidualBlock(64, 128, stride=2).cuda()
        f1, f2 = self.c3(fmap1[0]), self.c3(fmap2[0])
        f1, f2 = torch.cat([f1, fmap1[1]], dim=1), torch.cat([f2, fmap2[1]],
                                                             dim=1)
        f1, f2 = self.c2(f1), self.c2(f2)
        f1, f2 = torch.cat([f1, fmap1[2]], dim=1), torch.cat([f2, fmap2[2]],
                                                             dim=1)

        # all pairs correlation
        corr = CorrBlock.corr(f1, f2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = self._bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

    def _bilinear_sampler(self, img, coords, mode='bilinear', mask=False):
        """ Wrapper for grid_sample, uses pixel coordinates """
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    def _downsample_layer(self, levels=4, in_planes=64):
        """ Downsample feature map by levels """
        layers = []
        for _ in range(levels):
            layers.append(ResidualBlock(in_planes, 2 * in_planes, stride=2))
            in_planes *= 2
        layers.append(ResidualBlock(in_planes, in_planes, stride=1))
        return nn.Sequential(*layers)

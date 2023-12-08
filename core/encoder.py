import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianEncoder(nn.Module):

    def __init__(self, cfg):
        super(GaussianEncoder, self).__init__()

        self.convc1 = nn.Conv2d(4 * cfg.mixtures + 6, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2 * cfg.mixtures, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, cov, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(cov))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        print('cov', cov.shape)
        return torch.cat([out, cov], dim=1)  # 126+2*mixtures


class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=96,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96,
                            out_channels=96,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96,
                            out_channels=96,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96,
                            out_channels=128,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.netSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128,
                            out_channels=196,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196,
                            out_channels=196,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196,
                            out_channels=196,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

    # end

    def forward(self, tenInput):
        tenOne = self.netOne(tenInput)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = self.netFou(tenThr)
        tenFiv = self.netFiv(tenFou)
        tenSix = self.netSix(tenFiv)

        return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

    # end


# end
if __name__ == '__main__':
    input = torch.randn(1, 3, 1024, 436)
    encoder_layer = Encoder()
    output = encoder_layer(input)
    print('shape of output:')
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
    print(output[5].shape)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self,
                 enc_features=[96, 96, 192, 384, 2208],
                 decoder_width=0.5,
                 is_depthwise=False):
        super(Decoder, self).__init__()
        features = int(enc_features[-1] * decoder_width)

        padding = "zero"
        self.conv2 = Conv3x3(enc_features[-1], features, padding='zero')

        self.up1 = UpSampleBlock(skip_input=features // 1 + enc_features[-2],
                                 output_features=features // 2,
                                 padding=padding,
                                 is_depthwise=is_depthwise)
        self.up2 = UpSampleBlock(skip_input=features // 2 + enc_features[-3],
                                 output_features=features // 4,
                                 padding=padding,
                                 is_depthwise=is_depthwise)
        self.up3 = UpSampleBlock(skip_input=features // 4 + enc_features[-4],
                                 output_features=features // 8,
                                 padding=padding,
                                 is_depthwise=is_depthwise)
        self.up4 = UpSampleBlock(skip_input=features // 8 + enc_features[-5],
                                 output_features=features // 16,
                                 padding=padding,
                                 is_depthwise=is_depthwise)

        if not is_depthwise:
            self.conv3 = nn.Conv2d(features // 16,
                                   1,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   padding_mode='zeros')
        else:
            self.conv3 = Conv3x3(features // 16, 1, is_depthwise=is_depthwise)

    def forward(self, features):
        outputs = {}
        x_block0, x_block1, x_block2, x_block3, x_block4 = tuple(features)
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)

        outputs[("disp", 0)] = self.conv3(x_d4)
        return outputs


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 padding="zero",
                 stride=1,
                 is_depthwise=False):
        super(Conv3x3, self).__init__()

        if padding == "reflection":
            self.pad = nn.ReflectionPad2d(1)
        elif padding == "replicate":
            self.pad = nn.ReplicationPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)

        self.conv = nn.Conv2d(int(in_channels),
                              int(out_channels),
                              3,
                              stride=stride,
                              padding=0)  #padding_mode=padding)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode='nearest')


class UpSample(nn.Sequential):

    def __init__(self,
                 skip_input,
                 output_features,
                 align_corners=False,
                 padding="zero"):
        super(UpSample, self).__init__()
        self.convA = Conv3x3(skip_input, output_features, padding=padding)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = Conv3x3(output_features, output_features, padding=padding)
        self.leakyreluB = nn.LeakyReLU(0.2)

        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x,
                             size=[concat_with.size(2),
                                   concat_with.size(3)],
                             mode='nearest',
                             align_corners=self.align_corners)
        return self.leakyreluB(
            self.convB(
                self.leakyreluA(
                    self.convA(torch.cat([up_x, concat_with], dim=1)))))


class UpSampleBlock(nn.Sequential):

    def __init__(self,
                 skip_input,
                 output_features,
                 padding="zero",
                 is_depthwise=False):
        super(UpSampleBlock, self).__init__()
        self.convA = Conv3x3(skip_input,
                             output_features,
                             padding=padding,
                             is_depthwise=is_depthwise)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, concat_with):
        up_x = self.upsample(x)
        return self.leakyreluA(
            self.convA(torch.cat([up_x, concat_with], dim=1)))

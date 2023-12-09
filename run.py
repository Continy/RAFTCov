#!/usr/bin/env python
import argparse
import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
from core.network import Network
from configs.tartanair import get_cfg

torch.set_grad_enabled(False)

torch.backends.cudnn.enabled = True

arguments_strModel = 'default'  # 'default', or 'chairs-things'
arguments_strOne = './images/one.png'
arguments_strTwo = './images/two.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(
        sys.argv[1:], '',
    [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '':
        arguments_strModel = strArgument  # which model to use
    if strOption == '--one' and strArgument != '':
        arguments_strOne = strArgument  # path to the first frame
    if strOption == '--two' and strArgument != '':
        arguments_strTwo = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '':
        arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################

cfg = get_cfg()


##########################################################
def estimate(tenOne, tenTwo):
    global cfg

    netNetwork = Network(cfg).cuda().eval()

    assert (tenOne.shape[1] == tenTwo.shape[1])
    assert (tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert (intWidth == 1024)
    assert (intHeight == 436)

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(
        input=tenPreprocessedOne,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(
        input=tenPreprocessedTwo,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False)
    flow, cov = netNetwork(tenPreprocessedOne, tenPreprocessedTwo)
    tenFlow = torch.nn.functional.interpolate(input=flow,
                                              size=(intHeight, intWidth),
                                              mode='bilinear',
                                              align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()


# end

##########################################################

if __name__ == '__main__':
    tenOne = torch.FloatTensor(
        numpy.ascontiguousarray(
            numpy.array(
                PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(
                    2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(
        numpy.ascontiguousarray(
            numpy.array(
                PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(
                    2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    tenOutput = estimate(tenOne, tenTwo)

    objOutput = open(arguments_strOut, 'wb')

    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
    numpy.array([tenOutput.shape[2], tenOutput.shape[1]],
                numpy.int32).tofile(objOutput)
    numpy.array(tenOutput.numpy().transpose(1, 2, 0),
                numpy.float32).tofile(objOutput)

    objOutput.close()
# end

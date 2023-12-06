#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
from network import Network

##########################################################

torch.set_grad_enabled(
    False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

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

netNetwork = None

##########################################################
Network().load_state_dict({
    strKey.replace('module', 'net'): tenWeight
    for strKey, tenWeight in torch.hub.load_state_dict_from_url(
        url='http://content.sniklaus.com/github/pytorch-pwc/network-' +
        arguments_strModel + '.pytorch',
        file_name='pwc-' + arguments_strModel).items()
})


##########################################################
def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert (tenOne.shape[1] == tenTwo.shape[1])
    assert (tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert (
        intWidth == 1024
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert (
        intHeight == 436
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

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

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(
        tenPreprocessedOne, tenPreprocessedTwo),
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

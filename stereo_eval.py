import glob
import sys

sys.path.append('core')

from PIL import Image
import argparse
import os

import numpy as np
import torch
import cv2
from yacs.config import CfgNode as CN
from core.utils import vars_viz

from core.Stereo.network import RAFTCovWithStereoNet7 as Network
from core.utils.preprocess import build_cfg

import matplotlib.pyplot as plt
import torch.nn as nn
from core.utils.dis_viz import visualize_disparity_map
import torchvision.transforms.v2 as v2


def process_image(i, imgLlist, imgRlist, model, gt_stereo_npy, args):

    img1 = np.array(Image.open(imgLlist[i])).astype(np.uint8)
    img2 = np.array(Image.open(imgRlist[i])).astype(np.uint8)

    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img1 = img1.cuda()
    img2 = img2.cuda()
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    import core.utils.Utility as Utility

    gt_stereo = np.load(gt_stereo_npy)
    _, cropsize = Utility.getCropMargin(gt_stereo.shape)
    gt_stereo = torch.from_numpy(gt_stereo).unsqueeze(0).float()
    transform = v2.Compose([v2.CenterCrop(cropsize), v2.ToTensor()])
    gt_stereo = transform(gt_stereo).cuda()
    img1, img2 = img1 / 255, img2 / 255
    with torch.no_grad():
        stereo, covs = model(img1, img2)
    stereo = ((stereo / 320.0 / 0.25 / 0.02)) / 0.01
    gt_stereo = 1 / gt_stereo / 0.01
    cov = covs[-1]
    mse = torch.pow((stereo - gt_stereo), 2).squeeze_(0).cpu()
    mse = torch.mean(mse, dim=0)
    if args.cov:
        cov = torch.mean(cov, dim=1).cpu()
        cov.squeeze_(0)
        cov = cov.detach()
        cov = torch.exp(2 * cov)
        cov = torch.sqrt(cov)
        img = vars_viz.heatmap(cov)
        cv2.imwrite(result_path + 'cov/' + str(i).zfill(6) + '.png', img)
        os.makedirs(result_path + 'cov/' + 'file/', exist_ok=True)
        np.save(result_path + 'cov/' + 'file/' + str(i).zfill(6) + '.npy',
                cov.numpy())
        if not args.error and not args.mse:
            return 0
    if args.stereo:
        stereo = stereo.squeeze(0)
        stereo = stereo.cpu().detach().numpy()
        gt_stereo = gt_stereo.squeeze(0)
        gt_stereo = gt_stereo.cpu().detach().numpy()
        visualize_disparity_map(
            stereo,
            result_path + 'stereo/' + 'est/' + str(i).zfill(6) + '.png')
        visualize_disparity_map(
            gt_stereo,
            result_path + 'stereo/' + 'gt/' + str(i).zfill(6) + '.png')
        if not args.error and not args.mse:
            return 0
    if args.error or args.mse:

        mse = torch.sqrt(mse)
        img = vars_viz.heatmap(mse)
        cv2.imwrite(result_path + 'mse/' + str(i).zfill(6) + '.png', img)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/yaml/stereo.yaml')
    parser.add_argument('--stereo', action='store_true', default=True)
    parser.add_argument('--cov', action='store_true', default=True)
    parser.add_argument('--mse', action='store_true', default=True)
    parser.add_argument('--training_mode', default='cov')
    parser.add_argument('--error', default=False)
    args = parser.parse_args()
    cfg = build_cfg(args.config)
    cfg.update(vars(args))

    model_path = cfg.model_path
    img_path = cfg.img_path
    depth_path = cfg.depth_path

    imgLlist = glob.glob(img_path + 'image_left/*.png')
    imgLlist.sort()
    imgRlist = glob.glob(img_path + 'image_right/*.png')
    imgRlist.sort()
    depthlist = glob.glob(depth_path + '*.npy')
    depthlist.sort()

    length = cfg.length

    result_path = cfg.result_path

    for modelname in model_path:

        os.makedirs(result_path, exist_ok=True)
        os.makedirs(result_path + 'stereo/', exist_ok=True)
        os.makedirs(result_path + 'stereo/est/', exist_ok=True)
        os.makedirs(result_path + 'stereo/gt/', exist_ok=True)
        os.makedirs(result_path + 'cov/', exist_ok=True)
        os.makedirs(result_path + 'mse/', exist_ok=True)
        model = nn.DataParallel(Network(cfg))
        model.load_state_dict(torch.load(modelname), strict=False)

        model.cuda()
        model.eval()
        results = []
        for i in range(length):
            result = process_image(i, imgLlist, imgRlist, model, depthlist[i],
                                   args)
            print('{}/{}'.format(i + 1, length))
            if args.error or args.mse:
                results.append(result)

        if args.mse:
            results = np.array(results)
            mean = results.mean(axis=0)
            plt.plot(results, label='mse')
            plt.plot([mean] * length, label='mean')
            plt.savefig(result_path + 'mse.png')

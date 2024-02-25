import glob
import sys

from attr import validate

sys.path.append('core')
import torch.nn.functional as F
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import cv2
from yacs.config import CfgNode as CN
from core.utils import vars_viz
from core.utils import flow_viz
from core.network import RAFTCovWithPWCNet as Network
from configs.tartanair import get_cfg
from core.utils.preprocess import preprocess, reverse
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.stats import spearmanr


def process_image(i, filelist, model, gt_flow, args):

    img1 = np.array(Image.open(filelist[2 * i])).astype(np.uint8)
    img2 = np.array(Image.open(filelist[2 * i + 1])).astype(np.uint8)

    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    img1 = img1.cuda()
    img2 = img2.cuda()
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    img1, W, H, W_, H_ = preprocess(img1)
    img2, _, _, _, _ = preprocess(img2)
    with torch.no_grad():
        flow, covs = model(img1, img2)
    flow = reverse(flow, W, H, W_, H_, is_flow=True).squeeze_(0)
    covs = [reverse(cov, W, H, W_, H_) for cov in covs]
    cov = covs[-1]
    if args.flow:
        img = flow_viz.flow_to_image(flow.permute(1, 2, 0).cpu().numpy())
        image = Image.fromarray(img)
        image.save(result_path + 'flow/' + str(i).zfill(6) + '.png')
        if not args.error and not args.mse and not args.cov:
            return 0
    gt_flow = np.load(gt_flow[i])
    gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1).float()
    gt_flow = gt_flow.unsqueeze_(0).cuda()

    mse = torch.pow((flow - gt_flow), 2).squeeze_(0).cpu()
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
    if args.mse:
        mse = torch.sqrt(mse)
        img = vars_viz.heatmap(mse)
        cv2.imwrite(result_path + 'mse/' + str(i).zfill(6) + '.png', img)
        os.makedirs(result_path + 'mse/' + 'file/', exist_ok=True)
        np.save(result_path + 'mse/' + 'file/' + str(i).zfill(6) + '.npy',
                mse.numpy())
        return mse.mean()

    if args.error:
        cov = torch.mean(cov, dim=1).cpu().squeeze_(0).detach()
        cov = torch.mean(cov)
        mse = torch.mean(mse)
        print('mse:{}'.format(mse))
        return cov, mse
    return 0


if __name__ == '__main__':
    img_path = 'datasets/eval_data/img/'
    flow_path = 'datasets/eval_data/flow/'
    datapath = 'results/test/'
    i = list(range(5001, 210002, 5000))
    model_path = []
    for j in i:
        model_path.append('models/02_17_01_05/' + str(j) + '_RAFTCov.pth')
    # model_path.append('models/TartanAir.pth')
    imglist = glob.glob(img_path + '*.png')
    imglist.sort()
    flowlist = glob.glob(flow_path + '*.npy')
    flowlist.sort()
    length = len(flowlist)
    cfg = get_cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow', action='store_true', default=True)
    parser.add_argument('--cov', action='store_true', default=True)
    parser.add_argument('--mse', action='store_true', default=True)
    parser.add_argument('--training_mode', default='cov')
    parser.add_argument('--error', default=False)
    args = parser.parse_args()
    cfg.update(vars(args))
    for modelname in model_path:
        result_path = datapath + modelname.split('/')[2].split('_')[0] + '/'
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(result_path + 'flow/', exist_ok=True)
        os.makedirs(result_path + 'cov/', exist_ok=True)
        os.makedirs(result_path + 'mse/', exist_ok=True)
        model = nn.DataParallel(Network(cfg))
        model.load_state_dict(torch.load(modelname), strict=False)
        vonet_dict = torch.load('models/43_6_2_vonet_30000.pkl')
        new_state_dict = {}
        for key in vonet_dict.keys():
            if key.startswith('module.flowNet'):
                new_key = key.replace('module.flowNet', 'module.feature.pwc')
                new_state_dict[new_key] = vonet_dict[key]
            else:
                new_state_dict[key] = vonet_dict[key]
        model.load_state_dict(new_state_dict, strict=False)

        model.cuda()
        model.eval()
        results = []
        for i in range(length - 1):
            result = process_image(i, imglist, model, flowlist, args)
            print('{}/{}'.format(i + 1, length))
            if args.error or args.mse:
                results.append(result)

        if args.mse:
            results = np.array(results)
            mean = results.mean(axis=0)
            plt.plot(results, label='mse')
            plt.plot([mean] * length, label='mean')
            plt.savefig(result_path + 'mse.png')

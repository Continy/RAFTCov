from __future__ import print_function, division
import sys

sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2 as v2

torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader
from core import optimizer
import core.datasets as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.Flow.network import RAFTCovWithPWCNet as FlowNetwork
from core.Flow.network import PWCCov as CovNetwork
from core.Stereo.network import RAFTCovWithStereoNet7 as StereoNetwork
from core.utils.preprocess import *
import core.utils.Utility as Utility
# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:

        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


try:
    import wandb
except:
    print("wandb not installed, --wandb flag ignored")
    wandb = False
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]
#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    model = nn.DataParallel(CovNetwork(cfg))

    loguru_logger.info("Parameter Count: %d" %
                       count_parameters(model.module.netGaussian))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=False)

    vonet_dict = torch.load('models/43_6_2_vonet_30000.pkl')
    new_state_dict = {}
    for key in vonet_dict.keys():
        if key.startswith('module.flowNet'):
            new_key = key.replace('module.flowNet', 'module.feature.pwc')
            new_state_dict[new_key] = vonet_dict[key]
        else:
            new_state_dict[key] = vonet_dict[key]
    model.load_state_dict(new_state_dict, strict=False)
    for key in vonet_dict.keys():
        if key.startswith('module.flowNet'):
            new_key = key.replace('module.flowNet', 'module.netGaussian.pwc')
            new_state_dict[new_key] = vonet_dict[key]
        else:
            new_state_dict[key] = vonet_dict[key]
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.train()
    if cfg.training_mode == 'flow':
        #freeze the Covariance Decoder
        for param in model.module.netGaussian.parameters():
            param.requires_grad = False
        optimizer, scheduler = fetch_optimizer(model, cfg.trainer)
    if cfg.training_mode == 'cov':
        #freeze the FlowFormer
        for param in model.module.feature.pwc.parameters():
            param.requires_grad = False

    optimizer, scheduler = fetch_optimizer(model, cfg)
    train_loader = datasets.fetch_dataloader(cfg)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    if cfg.log:
        logger = Logger(model, scheduler, cfg)
    if cfg.wandb and wandb:
        wandb.init(project='RAFTCov', name=cfg.name, config=cfg)
        wandb.watch(model, log='all')
    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
            image1, image2, gt_flow, valid = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 +
                          stdv * torch.randn(*image1.shape).cuda()).clamp(
                              0.0, 255.0)
                image2 = (image2 +
                          stdv * torch.randn(*image2.shape).cuda()).clamp(
                              0.0, 255.0)

            output = {}
            image1, shapes = preprocess(image1)
            image2, _ = preprocess(image2)
            #with torch.autocast('cuda'):
            flow, covs = model(image1, image2)

            loss, metrics = sequence_loss(cfg=cfg,
                                          preds=flow,
                                          targets=gt_flow,
                                          valid=valid,
                                          cov_preds=covs,
                                          shapes=shapes)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            if cfg.log:
                metrics.update(output)
                logger.push(metrics)

            if total_steps % 5 == 0:
                print("Iter: %d, Loss: %.4f" % (total_steps, loss.item()))
                if cfg.wandb and wandb:
                    wandb.log(metrics)

            total_steps += 1

            if total_steps > cfg.num_steps:
                should_keep_training = False
                break
            if cfg.autosave_freq and total_steps % cfg.autosave_freq == 0:
                if cfg.log or cfg.wandb:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1,
                                             cfg.name)
                    torch.save(model.state_dict(), PATH)
    if cfg.log:
        logger.close()
        PATH = cfg.log_dir + '/final'
        torch.save(model.state_dict(), PATH)

    PATH = f'models/{cfg.savename}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


def train_stereo(cfg):
    model = nn.DataParallel(StereoNetwork(cfg))

    loguru_logger.info("Parameter Count: %d" %
                       count_parameters(model.module.decoder))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=False)

    vonet_dict = torch.load('models/43_6_2_vonet_30000.pkl')
    new_state_dict = {}
    for key in vonet_dict.keys():
        if key.startswith('module.stereoNet'):
            new_key = key.replace('module.stereoNet', 'module.feature.stereo')
            new_state_dict[new_key] = vonet_dict[key]
        else:
            new_state_dict[key] = vonet_dict[key]
    model.load_state_dict(new_state_dict, strict=False)
    new_state_dict = {}

    for key in vonet_dict.keys():
        if key.startswith('module.stereoNet'):
            new_key = key.replace('module.stereoNet', 'module.decoder')
            new_state_dict[new_key] = vonet_dict[key]
        else:
            new_state_dict[key] = vonet_dict[key]
    model.load_state_dict(new_state_dict, strict=False)

    model.cuda()
    model.train()
    if cfg.training_mode == 'stereo':
        #freeze the Covariance Decoder
        for param in model.module.netGaussian.parameters():
            param.requires_grad = False
        optimizer, scheduler = fetch_optimizer(model, cfg.trainer)
    if cfg.training_mode == 'cov':
        for param in model.module.feature.stereo.parameters():
            param.requires_grad = False

    optimizer, scheduler = fetch_optimizer(model, cfg)
    train_loader = datasets.fetch_dataloader(cfg)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    if cfg.log:
        logger = Logger(model, scheduler, cfg)
    if cfg.wandb and wandb:
        wandb.init(project=cfg.project, name=cfg.name, config=cfg)
        wandb.watch(model, log=None)
    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
            image1, image2, gt_stereo = [x for x in data_blob]
            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape)).clamp(
                    0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape)).clamp(
                    0.0, 255.0)

            output = {}
            image1 = image1 / 255.0
            image2 = image2 / 255.0

            flow, covs = model(image1, image2)
            _, cropsize = Utility.getCropMargin(gt_stereo.shape)
            transform = v2.Compose([v2.CenterCrop(cropsize), v2.ToTensor()])
            gt_stereo = transform(gt_stereo).cuda()

            loss, metrics = sequence_loss(cfg=cfg,
                                          preds=flow,
                                          targets=gt_stereo,
                                          valid=None,
                                          cov_preds=covs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)

            scaler.step(optimizer)
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            metrics['lr'] = lr
            scaler.update()
            if cfg.log:
                metrics.update(output)
                logger.push(metrics)

            if total_steps % 5 == 0:
                print("Iter: %d, Loss: %.4f" % (total_steps, loss.item()))
                if cfg.wandb and wandb:
                    wandb.log(metrics)

            total_steps += 1

            if total_steps > cfg.num_steps:
                should_keep_training = False
                break
            if cfg.autosave_freq and total_steps % cfg.autosave_freq == 0:
                if cfg.log or cfg.wandb:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1,
                                             cfg.name)
                    torch.save(model.state_dict(), PATH)
    if cfg.log:
        logger.close()
        PATH = cfg.log_dir + '/final'
        torch.save(model.state_dict(), PATH)

    PATH = f'models/{cfg.savename}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default='RAFTCov',
                        help="name your experiment")
    parser.add_argument('--log',
                        action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='enable wandb logging')
    parser.add_argument('--config', default='configs/train/stereo_small.yaml')
    parser.add_argument('--exp',
                        action='store_true',
                        help='Using exp() for activation function')
    parser.add_argument('--without_mask',
                        action='store_true',
                        help='without motion mask')
    parser.add_argument('--eps',
                        action='store_true',
                        help='add epsilon to the cov loss')
    args = parser.parse_args()
    cfg = build_cfg(args.config)
    cfg.update(vars(args))

    if args.log or args.wandb:
        process_cfg(cfg)
        loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
        cfg.log = True
    #loguru_logger.info(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs('models', exist_ok=True)
    if cfg.stereo:
        train_stereo(cfg)
    else:
        train(cfg)

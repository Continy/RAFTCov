import torch
import cv2
from PIL import Image
from utils import flow_viz
from core.utils.preprocess import *

EPSILON = 1e-3


def flow_loss(cfg, flow_pred, flow_gt, valid, cov_preds, shape):

    flow = reverse(flow_pred, shape, is_flow=True)
    covs = [reverse(cov, shape) for cov in cov_preds]
    gamma = cfg.gamma
    max_cov = cfg.max_cov
    n_predictions = len(cov_preds)

    mse_loss = torch.zeros_like(flow_gt)
    cov_loss = torch.zeros_like(flow_gt)
    mse_loss = (flow - flow_gt)**2

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_cov)
    if not cfg.without_mask:
        mse_loss = (valid[:, None] * mse_loss)
    #mse_loss = torch.mean(mse_loss, dim=1)
    covs = [
        cov.view(cov.shape[0], 2, cov.shape[1] // 2, cov.shape[2],
                 cov.shape[3]) for cov in covs
    ]
    cov_preds = [cov.mean(dim=2) for cov in covs]
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (mse_loss /
                  (2 * torch.exp(2 * cov_preds[i])) + cov_preds[i]) * i_weight
        if not cfg.without_mask:
            i_loss = valid[:, None] * i_loss
        cov_loss += i_loss

    cov = torch.exp(2 * torch.mean(torch.stack(cov_preds, dim=0), dim=0))

    metrics = {
        'sqrt_cov': cov.sqrt().float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return cov_loss.mean(), metrics


def simple_flow_loss(cfg, flow_pred, flow_gt, valid, cov_preds, shape):
    flow = reverse(flow_pred, shape, is_flow=True)
    covs = reverse(cov_preds, shape)
    gamma = cfg.gamma
    max_cov = cfg.max_cov

    mse_loss = torch.zeros_like(flow_gt)
    cov_loss = torch.zeros_like(flow_gt)
    mse_loss = (flow - flow_gt)**2

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_cov)
    if not cfg.without_mask:
        mse_loss = (valid[:, None] * mse_loss)
    #mse_loss = torch.mean(mse_loss, dim=1)

    cov_loss = (mse_loss / (2 * torch.exp(2 * covs)) + covs)

    cov = torch.exp(2 * torch.mean(cov_preds, dim=0))

    metrics = {
        'sqrt_cov': cov.sqrt().float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return cov_loss.mean(), metrics


def stereo_loss(cfg, stereo_pred, depth_gt, valid, cov_preds):
    # For TartanAir, fx = 320.0, baseline = 0.25
    # depth = (fx * baseline) / (stereo_pred + 1e-8)
    # cov_preds : stereo covariance predictions
    # stereo_pred : disparity predictions
    # stereo_gt = (fx * baseline) / (depth_gt + 1e-8)
    # 0.02:stereo normalization factor, 0.01:resize factor

    gamma = cfg.gamma
    max_cov = cfg.max_cov
    n_predictions = len(cov_preds)

    mse_loss = torch.zeros_like(depth_gt)
    cov_loss = torch.zeros_like(depth_gt)

    depth_est = ((stereo_pred / 320.0 / 0.25 / 0.02)) / 0.01
    depth_gt = 1 / depth_gt / 0.01
    mse_loss = (depth_est - depth_gt)**2
    mag = torch.sum(depth_est**2, dim=1).sqrt()
    valid = (mag < max_cov)
    if not cfg.without_mask:
        mse_loss = (valid[:, None] * mse_loss)

    cov_preds = [
        cov.view(cov.shape[0], 1, cov.shape[1], cov.shape[2], cov.shape[3])
        for cov in cov_preds
    ]
    cov_preds = [cov.mean(dim=2) for cov in cov_preds]
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (mse_loss /
                  (2 * torch.exp(2 * cov_preds[i])) + cov_preds[i]) * i_weight
        if not cfg.without_mask:
            i_loss = valid[:, None] * i_loss
        cov_loss += i_loss

    cov = torch.exp(2 * torch.mean(torch.stack(cov_preds, dim=0), dim=0))

    metrics = {
        'sqrt_cov': cov.sqrt().float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return cov_loss.mean(), metrics


def simple_stereo_loss(cfg, stereo_pred, depth_gt, valid, cov_preds):

    mse_loss = torch.zeros_like(depth_gt)
    cov_loss = torch.zeros_like(depth_gt)

    depth_est = ((stereo_pred / 320.0 / 0.25 / 0.02)) / 0.01
    depth_gt = 1 / depth_gt / 0.01
    mse_loss = (depth_est - depth_gt)**2

    mag = torch.sum(depth_est**2, dim=1).sqrt()
    valid = (mag < cfg.max_cov)
    if not cfg.without_mask:
        mse_loss = (valid[:, None] * mse_loss)
    if cfg.epsilon:
        cov_loss = (mse_loss /
                    (cov_preds + EPSILON)) + torch.log(cov_preds + EPSILON)
    else:
        cov_loss = (mse_loss / (cov_preds)) + torch.log(cov_preds)
    metrics = {
        'sqrt_cov': cov_preds.sqrt().float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return cov_loss.mean(), metrics


def sequence_loss(**kwargs):
    cfg, preds, targets, valid, cov_preds = kwargs['cfg'], kwargs[
        'preds'], kwargs['targets'], kwargs['valid'], kwargs['cov_preds']

    if cfg.stereo:
        if cfg.decoder == 'hourglass':
            return simple_stereo_loss(cfg, preds, targets, valid, cov_preds)
        elif cfg.decoder == 'attention':
            return stereo_loss(cfg, preds, targets, valid, cov_preds)
        else:
            raise NotImplementedError
    else:
        if cfg.decoder == 'pwc':
            return simple_flow_loss(cfg,
                                    preds,
                                    targets,
                                    valid,
                                    cov_preds,
                                    shape=kwargs['shapes'])
        elif cfg.decoder == 'raft':
            return flow_loss(cfg,
                             preds,
                             targets,
                             valid,
                             cov_preds,
                             shape=kwargs['shapes'])
        else:
            raise NotImplementedError

import torch
import cv2
from PIL import Image
from utils import flow_viz


def flow_loss(flow_pred, flow_gt, valid, cfg, cov_preds, without_mask=False):

    gamma = cfg.gamma
    max_cov = cfg.max_cov
    n_predictions = len(cov_preds)

    mse_loss = torch.zeros_like(flow_gt)
    cov_loss = torch.zeros_like(flow_gt)
    mse_loss = (flow_pred - flow_gt)**2

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_cov)
    if not without_mask:
        mse_loss = (valid[:, None] * mse_loss)
    #mse_loss = torch.mean(mse_loss, dim=1)
    cov_preds = [
        cov.view(cov.shape[0], 2, cov.shape[1] // 2, cov.shape[2],
                 cov.shape[3]) for cov in cov_preds
    ]
    cov_preds = [cov.mean(dim=2) for cov in cov_preds]
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (mse_loss /
                  (2 * torch.exp(2 * cov_preds[i])) + cov_preds[i]) * i_weight
        if not without_mask:
            i_loss = valid[:, None] * i_loss
        cov_loss += i_loss

    cov = torch.exp(2 * torch.mean(torch.stack(cov_preds, dim=0), dim=0))

    metrics = {
        'sqrt_cov': cov.sqrt().float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return cov_loss.mean(), metrics


def stereo_loss(stereo_pred,
                depth_gt,
                valid,
                cfg,
                cov_preds,
                without_mask=False):
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
    if not without_mask:
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
        if not without_mask:
            i_loss = valid[:, None] * i_loss
        cov_loss += i_loss

    cov = torch.exp(2 * torch.mean(torch.stack(cov_preds, dim=0), dim=0))

    metrics = {
        'sqrt_cov': cov.sqrt().float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return cov_loss.mean(), {}


def sequence_loss(*args):
    cfg = args[3]
    if cfg.stereo:
        return stereo_loss(*args)
    else:
        return flow_loss(*args)

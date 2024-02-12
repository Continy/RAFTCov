import torch
import sys
import cv2
from utils.flow_viz import flow_to_image
from PIL import Image


def sequence_loss(flow_pred, flow_gt, valid, cfg, cov_preds, method='norm'):

    gamma = cfg.gamma
    max_cov = cfg.max_cov
    n_predictions = len(cov_preds)
    flow_gt_thresholds = [5, 10, 20]

    mse_loss = torch.zeros_like(flow_gt)
    cov_loss = torch.zeros_like(flow_gt)

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 50)

    mse_loss = (flow_pred - flow_gt)**2
    mse_loss = (valid[:, None] * mse_loss)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = mse_loss / (2 * torch.exp(2 * cov_preds[i])) + cov_preds[i]
        cov_loss += valid[:, None] * i_weight * i_loss

    cov = torch.exp(2 * torch.mean(torch.stack(cov_preds, dim=0), dim=0))
    if cfg.training_viz:
        viz = cov.mean(dim=0).mean(
            dim=0).squeeze_(0).squeeze_(0).detach().cpu()
        from utils.vars_viz import heatmap
        cv2.imwrite('cov.png', heatmap(viz))
        mse = mse_loss.mean(dim=0).mean(
            dim=0).squeeze_(0).squeeze_(0).detach().cpu()
        cv2.imwrite('mse.png', heatmap(mse))

    metrics = {
        'cov': cov.float().mean().item(),
        'sqrt_cov': cov.sqrt().float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return cov_loss.mean(), metrics

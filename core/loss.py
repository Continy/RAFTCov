import torch
import sys
import cv2


def sequence_loss(flow_preds, flow_gt, valid, cfg, cov_preds):

    gamma = cfg.gamma
    max_cov = cfg.max_cov
    n_predictions = len(flow_preds)
    flow_gt_thresholds = [5, 10, 20]

    mse_loss = torch.zeros_like(flow_gt)
    cov_loss = torch.zeros_like(flow_gt)
    distribution_loss = torch.zeros_like(flow_gt)

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_cov)

    i_loss = (flow_preds - flow_gt)**2
    mse_loss += (valid[:, None] * i_loss)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = mse_loss / (2 * torch.exp(2 * cov_preds[i])) + cov_preds[i]
        cov_loss += (valid[:, None] * i_weight * i_loss)
    mse_loss = torch.mean(mse_loss, dim=1)
    cov = torch.exp(2 * torch.mean(torch.stack(cov_preds, dim=0), dim=0))
    if cfg.training_viz:
        viz = torch.mean(
            cov, dim=0).squeeze_(0).squeeze_(0).detach().cpu().numpy() * 255
        cv2.imwrite('cov.png', viz)
    metrics = {
        'cov': cov.float().mean().item(),
        'cov_loss': cov_loss.float().mean().item(),
        'mse_loss': mse_loss.float().mean().item()
    }
    return distribution_loss.mean(), metrics

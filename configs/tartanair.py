from yacs.config import CfgNode as CN

cfg = CN()

#paths:
cfg.stage = 'tartanair'
cfg.image_size = [480, 640]
cfg.root = 'D:\\gits\\FlowFormer\\datasets\\abandonedfactory\\Easy\\P001\\'
cfg.save_path = 'results/P001/'
cfg.restore_ckpt = 'models/default'
cfg.folderlength = 1
cfg.training_mode = 'cov'
cfg.seed = 1234
cfg.log = False

#gaussian:
cfg.dim = 32
cfg.dropout = 0.1

cfg.num_heads = 2
cfg.mixtures = 4
cfg.gru_iters = 12

#training:
cfg.mixed_precision = True
cfg.optimizer = 'adamw'
cfg.scheduler = 'OneCycleLR'
cfg.add_noise = True
cfg.canonical_lr = 12.5e-4
cfg.adamw_decay = 1e-4
cfg.num_steps = 2000
cfg.epsilon = 1e-8
cfg.anneal_strategy = 'linear'
cfg.batch_size = 4
cfg.num_workers = 0
cfg.autosave_freq = 500
cfg.training_viz = False
cfg.clip = 1.0

#loss
cfg.gamma = 0.85
cfg.max_cov = 50


def get_cfg():
    return cfg.clone()

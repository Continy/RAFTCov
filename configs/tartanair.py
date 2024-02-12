from yacs.config import CfgNode as CN

cfg = CN()

#paths:
cfg.stage = 'tartanair'
cfg.image_size = [480, 640]
cfg.root = 'D:\\gits\\FlowFormer\\datasets\\abandonedfactory\\Easy\\P001\\'
cfg.save_path = 'results/P001/'
cfg.savename = 'TartanAir'
#cfg.restore_ckpt = 'models/60001_flowformer.pth'
cfg.restore_ckpt = None
cfg.tartanvo_model = 'models/tartanvo_1914.pkl'
cfg.folderlength = 1
cfg.training_mode = 'cov'
cfg.seed = 2568
cfg.log = False
cfg.sum_freq = 100
#gaussian:
cfg.dim = 128
cfg.dropout = 0.1

cfg.num_heads = 4
cfg.mixtures = 4
cfg.gru_iters = 12

#training:
cfg.mixed_precision = True
cfg.optimizer = 'adamw'
cfg.scheduler = 'OneCycleLR'
cfg.add_noise = True
cfg.canonical_lr = 12.5e-5
cfg.adamw_decay = 1e-5
cfg.num_steps = 120000
cfg.epsilon = 1e-8
cfg.anneal_strategy = 'linear'
cfg.batch_size = 8
cfg.num_workers = 0
cfg.autosave_freq = 5000
cfg.training_viz = False
cfg.clip = 1.2
cfg.loss_method = 'mean'

#loss
cfg.gamma = 0.85
cfg.max_cov = 200


def get_cfg():
    return cfg.clone()

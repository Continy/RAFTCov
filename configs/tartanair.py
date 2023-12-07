from yacs.config import CfgNode as CN

cfg = CN()
#paths:
cfg.dataset_path = '/home/zhengwu/data/tartanair_release1/abandonedfactory'
cfg.save_path = 'results/P001/'
cfg.dataset_name = 'tartanair'
cfg.restore_ckpt = 'models/default'

#gaussian:
cfg.dim = 128
cfg.dropout = 0.1
cfg.hidden_dim = 128
cfg.num_heads = 4
cfg.mixtures = 5
cfg.gru_iters = 12

#training:
cfg.scheduler = 'OneCycleLR'
cfg.optimizer = 'adamw'
cfg.batch_size = 4
cfg.epochs = 50000
cfg.lr = 0.0001
cfg.num_workers = 0
cfg.save_freq = 500
cfg.weight_decay = 0.0001

#loss
cfg.gamma = 0.85
cfg.max_cov = 50


def get_cfg():
    return cfg.clone()

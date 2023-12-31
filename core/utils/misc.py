import time
import os


def process_cfg(cfg):
    log_dir = 'logs/' + cfg.name + '/'

    now = time.localtime()
    now_time = '{:02d}_{:02d}_{:02d}_{:02d}'.format(now.tm_mon, now.tm_mday,
                                                    now.tm_hour, now.tm_min)
    log_dir += now_time
    cfg.log_dir = log_dir
    if os.path.exists(log_dir):
        os.system('rm -rf ' + log_dir)
    os.makedirs(log_dir)

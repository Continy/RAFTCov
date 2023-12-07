# Using pyyaml to read configs

The .yaml file's format is as follows:

```yaml
#tartanair config:
dataset_path: 'where/you/put/the/dataset'
save_path: 'where/you/want/to/save/the/result'
dataset_name: 'your_dataset_name'
restore_ckpt: None

gaussian:
  att_dim: # attention dimension
  dropout: # dropout rate
  hidden_dim: # hidden dimension of the GRU
  num_heads: # number of heads in multi-head attention
  gru_iters: # number of iterations of the GRU

train:
  #training configs
  scheduler: 'OneCycleLR'
  optimizer: 'adamw'
  batch_size: 4
  epochs: 50000
  lr: 0.0001
  num_workers: 0
  save_freq: 500
  weight_decay: 0.0001
```

usage:

```python
import yaml
def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

config = read_config('configs/config.yaml')
# config['train']['batch_size']
```

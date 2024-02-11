pip install wandb
export WANDB_API_KEY='need to be filled'
wandb login $WANDB_API_KEY
python -u /zihao/RAFTCov/train.py --log 
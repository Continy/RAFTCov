pip install wandb
pip install --upgrade timm
export WANDB_API_KEY='need to be filled'
wandb login $WANDB_API_KEY
python -u /zihao/RAFTCov/train.py --wandb --stereo --config --exp configs/train/flow_hourglass.yaml
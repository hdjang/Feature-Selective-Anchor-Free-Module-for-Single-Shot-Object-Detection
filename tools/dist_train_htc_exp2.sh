#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
DIRNAME=./tools/dist_train_htc_exp2.sh
CONFIG=./configs/htc/htc_r50_fpn_1x_exp2.py
GPUS=4

CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 1001 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

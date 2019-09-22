#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
DIRNAME=./tools/dist_train_htc_exp3.sh
CONFIG=./configs/htc/htc_r34_fpn_1x_exp3.py
GPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 1002 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

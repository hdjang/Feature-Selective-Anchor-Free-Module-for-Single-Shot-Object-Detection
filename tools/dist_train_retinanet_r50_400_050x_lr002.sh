#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_400_050x_lr002.py
GPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 1002 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

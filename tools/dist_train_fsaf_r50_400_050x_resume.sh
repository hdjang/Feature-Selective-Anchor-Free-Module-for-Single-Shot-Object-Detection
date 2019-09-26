#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/fsaf_r50_400_050x.py
GPUS=4
RESUME=./work_dirs/fsaf_r50_400_050x/epoch_2.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 1005 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --resume_from $RESUME
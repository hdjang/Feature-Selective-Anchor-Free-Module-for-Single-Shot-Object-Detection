#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
DIRNAME=./tools/dist_train_retina_exp2.sh
CONFIG=./configs/retinanet_r34_fpn_1x_exp2.py
GPUS=4

CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 1001 \
    $(dirname "$DIRNAME")/train.py $CONFIG --launcher pytorch ${@:3}

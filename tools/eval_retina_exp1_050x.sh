#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_400_050x.py
CKPT=/rcv/user/workspace/mmdetection/work_dirs/retinanet_r50_fpn_050x_exp1/epoch_1.pth
OUT=/rcv/user/workspace/mmdetection/work_dirs/retinanet_r50_fpn_050x_exp1/results_epoch_1.pkl

CUDA_VISIBLE_DEVICES=5 $PYTHON $(dirname "$0")/test.py \
                                               $CONFIG     \
                                               $CKPT       \
                                               --out  $OUT \
                                               --eval bbox \
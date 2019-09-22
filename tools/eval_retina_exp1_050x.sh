#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_fpn_050x_exp1.py
CKPT=/rcv/user/workspace/mmdetection/work_dirs/retinanet_r50_fpn_050x_exp1/epoch_6.pth
OUT=/rcv/user/workspace/mmdetection/work_dirs/retinanet_r50_fpn_050x_exp1/results_epoch_6.pkl

CUDA_VISIBLE_DEVICES=0 $PYTHON $(dirname "$0")/test.py \
                                               $CONFIG     \
                                               $CKPT       \
                                               --out  $OUT \
                                               --eval bbox \
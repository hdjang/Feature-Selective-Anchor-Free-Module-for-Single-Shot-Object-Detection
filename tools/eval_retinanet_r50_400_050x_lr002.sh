#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_400_050x_lr002.py
CKPT=./models/retinanet_r50_400_050x_lr002.pth
OUT=./models/results_retinanet_r50_400_050x_lr002.pkl

CUDA_VISIBLE_DEVICES=0 $PYTHON $(dirname "$0")/test.py \
                                               $CONFIG     \
                                               $CKPT       \
                                               --out  $OUT \
                                               --eval bbox \
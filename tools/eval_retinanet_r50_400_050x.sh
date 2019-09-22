#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_400_050x.py
CKPT=./models/retinanet_r50_400_050x.pth
OUT=./models/results_retinanet_r50_400_050x.pkl

CUDA_VISIBLE_DEVICES=0 $PYTHON $(dirname "$0")/test.py \
                                               $CONFIG     \
                                               $CKPT       \
                                               --out  $OUT \
                                               --eval bbox \
#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/fsaf_r50_400_050x.py
CKPT=./models/fsaf_r50_400_050x.pth
OUT=./models/results_fsaf_r50_400_050x.pkl

CUDA_VISIBLE_DEVICES=1 $PYTHON $(dirname "$0")/test.py \
                                               $CONFIG     \
                                               $CKPT       \
                                               --out  $OUT \
                                               --eval bbox \
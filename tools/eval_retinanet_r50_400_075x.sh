#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_400_075x.py
CKPT=./models/retinanet_r50_400_075x.pth
OUT=./models/resultsnet_retina_r50_400_075x.pkl

CUDA_VISIBLE_DEVICES=1 $PYTHON $(dirname "$0")/test.py \
                                               $CONFIG     \
                                               $CKPT       \
                                               --out  $OUT \
                                               --eval bbox \
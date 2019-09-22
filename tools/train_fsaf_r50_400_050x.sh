#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/fsaf_r50_400_050x.py

CUDA_VISIBLE_DEVICES=0 $PYTHON $(dirname "$0")/train.py $CONFIG
#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/fsaf_r50_400.py

CUDA_VISIBLE_DEVICES=1 $PYTHON $(dirname "$0")/train.py $CONFIG
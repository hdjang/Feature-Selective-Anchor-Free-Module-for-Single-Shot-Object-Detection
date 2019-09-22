#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/fsaf_r50_fpn_050x_exp1.py

CUDA_VISIBLE_DEVICES=2 $PYTHON $(dirname "$0")/train.py $CONFIG
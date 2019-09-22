#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_fpn_050x_exp1.py

CUDA_VISIBLE_DEVICES=3 $PYTHON $(dirname "$0")/train.py $CONFIG
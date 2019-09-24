#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=./configs/retinanet_r50_fpn_050x_exp1.py

CUDA_VISIBLE_DEVICES=6 $PYTHON $(dirname "$0")/train.py $CONFIG
#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train_ffwm.py \
                --name ffwm  \
                --preload \
                --dataroot  /opt/data/private/dataset/ \
                --lightcnn ./checkpoints/lightCNN_10_checkpoint.pth
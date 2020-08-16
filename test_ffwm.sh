#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python test_ffwm.py \
                --dataroot /opt/data/private/dataset \
                --lightcnn ./checkpoints/lightCNN_10_checkpoint.pth \
                --name ffwm \
                --preload
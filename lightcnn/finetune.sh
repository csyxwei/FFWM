#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python finetune.py \
                --save_path ../checkpoints/ \
                --dataroot /opt/data/private/dataset/multipie \
                --model_path ../checkpoints/LightCNN_29Layers_checkpoint.pth \
                --preload
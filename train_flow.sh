#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train_flow.py \
                --model flownet \
                --dataroot /opt/data/private/dataset/ \
                --reverse \
                --aug \
                --preload \
                --name flownetb \
                --batch_size 6
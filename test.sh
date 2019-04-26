#!/bin/bash

GPU=$1
GROUP=$2
SCALE=$3

CUDA_VISIBLE_DEVICES=${GPU} python test.py --net vgg16 --lr 0.01 --group ${GROUP} --scale ${SCALE}
CUDA_VISIBLE_DEVICES=${GPU} python test.py --net vgg16 --lr 0.005 --group ${GROUP} --scale ${SCALE}
CUDA_VISIBLE_DEVICES=${GPU} python test.py --net vgg16 --lr 0.001 --group ${GROUP} --scale ${SCALE}
CUDA_VISIBLE_DEVICES=${GPU} python test.py --net vgg16 --lr 0.0005 --group ${GROUP} --scale ${SCALE}

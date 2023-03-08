#!/bin/bash

source /home/ubuntu/.pyenv/versions/venv.cnet/bin/activate

PROJECT_PATH="/home/ubuntu/cultionet_train"
MODEL_TYPE="ResUNet3Psi"
RES_BLOCK_TYPE="res"

cultionet train -p $PROJECT_PATH --val-frac 0.1 --epochs 10 \
	--deep-sup-dist --deep-sup-edge --deep-sup-mask \
	--learning-rate 1e-3 --scale-pos-weight --model-type $MODEL_TYPE \
	--res-block-type $RES_BLOCK_TYPE --dilations 2 \
	--attention-weights spatial_channel --batch-size 8 \
	--refine-calibrate --skip-train

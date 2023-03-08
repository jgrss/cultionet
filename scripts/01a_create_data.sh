#!/bin/bash

source /home/ubuntu/.pyenv/versions/venv.cnet/bin/activate

PROJECT_PATH="/home/ubuntu/cultionet_train"
CROP_COLUMN=""
REPLACE_DICT=""

cultionet create -p $PROJECT_PATH -gs 100 100 --crop-column $CROP_COLUMN \
	--max-crop-class 1 --replace-dict $REPLACE_DICT \
	--feature-pattern {region}/{image_vi} \
	--image-date-format %Y%m%d \
	--config-file ${PROJECT_PATH}/config.yml

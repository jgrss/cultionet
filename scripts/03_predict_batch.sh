#!/bin/bash

PROJECT_PATH="/home/ubuntu/cultionet_train"
BATCH_SIZE=8
YEAR=2022
MODEL_TYPE="ResUNet3Psi"
RES_BLOCK_TYPE="res"

source /home/ubuntu/.pyenv/versions/venv.cnet/bin/activate

declare -a REGIONS=("")
for REGION in ${REGIONS[@]}; do
  echo $REGION
  cultionet predict \
  	  -p $PROJECT_PATH -y $YEAR -o ${PROJECT_PATH}/predictions/${REGION}.tif \
  	  -d ${PROJECT_PATH}/data/predict/processed/ --region $REGION \
  	  --ref-image ${PROJECT_PATH}/${REGION}/brdf_ts/ms/evi2/20210101.tif \
  	  --batch-size $BATCH_SIZE --config-file ${PROJECT_PATH}/config.yml \
	  --num-classes 2 --model-type $MODEL_TYPE --res-block-type $RES_BLOCK_TYPE \
	  --dilations 2 --attention-weights spatial_channel

done


#!/bin/bash

PROJECT_PATH="/home/ubuntu/cultionet_train"
CHUNKSIZE=200
NUM_WORKERS=6

source /home/ubuntu/.pyenv/versions/venv.cnet/bin/activate

declare -a REGIONS=("")
for REGION in ${REGIONS[@]}; do
  echo $REGION
  cultionet create-predict \
	  -p $PROJECT_PATH -y 2022 --ts-path ${PROJECT_PATH}/${REGION} \
	  --chunksize $CHUNKSIZE --append-ts n \
	  --image-date-format %Y%m%d -n $NUM_WORKERS \
	  --config-file ${PROJECT_PATH}/config.yml
done

#!/usr/bin/env bash

OUTPUT_DIR=/tmp/tfrecords_20190119

if [[ ! -d ${OUTPUT_DIR} ]]; then
    mkdir ${OUTPUT_DIR}
fi

# Data convertion for Pascal VOC 2007
PASCAL_VOC_2007=./data/VOC2007/train/
python3 tf_convert_data.py  \
    --dataset_name=pascalvoc    \
    --dataset_dir=${PASCAL_VOC_2007}    \
    --output_name=voc_2007_train    \
    --output_dir=${OUTPUT_DIR}


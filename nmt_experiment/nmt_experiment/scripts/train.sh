#!/bin/bash

project_dir=/mnt/d/file/nlp/exp3/nmt_experiment/nmt_experiment

export PYTHONPATH=${project_dir}:${PYTHONPATH}

python3 -u ${project_dir}/nmt/train.py \
    --config-path ${project_dir}/config.yaml

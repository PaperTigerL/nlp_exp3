#!/bin/bash

project_dir=/mnt/d/file/nlp/exp3/nmt_experiment/nmt_experiment

spm_train \
    --input ${project_dir}/data/raw_corpus/train.zh-en.zh \
    --model_prefix ${project_dir}/data/spm_data/spm.zh \
    --vocab_size 8000 \
    --character_coverage 1.0 \
    --model_type bpe

spm_train \
    --input ${project_dir}/data/raw_corpus/train.zh-en.en \
    --model_prefix ${project_dir}/data/spm_data/spm.en \
    --vocab_size 8000 \
    --character_coverage 1.0 \
    --model_type bpe

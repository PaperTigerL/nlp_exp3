#!/bin/bash

project_dir=/mnt/d/file/nlp/exp3/nmt_experiment/nmt_experiment

python3 -u ${project_dir}/nmt/build_vocab.py \
    --corpus-path ${project_dir}/data/spm_corpus/train.bpe.zh-en.zh \
    --vocab-path ${project_dir}/data/spm_corpus/vocab.zh.json \

python3 -u ${project_dir}/nmt/build_vocab.py \
    --corpus-path ${project_dir}/data/spm_corpus/train.bpe.zh-en.en \
    --vocab-path ${project_dir}/data/spm_corpus/vocab.en.json \

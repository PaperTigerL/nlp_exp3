#!/bin/bash

project_dir=/mnt/d/file/nlp/exp3/nmt_experiment/nmt_experiment

spm_corpus_dir=${project_dir}/data/spm_corpus
mkdir -p ${spm_corpus_dir}

for lang in "zh" "en"; do
    for file_type in "train" "valid" "test"; do
        spm_encode \
            --model ${project_dir}/data/spm_data/spm.${lang}.model \
            --output_format piece \
            --input ${project_dir}/data/raw_corpus/${file_type}.zh-en.${lang} \
            --output ${spm_corpus_dir}/${file_type}.bpe.zh-en.${lang}
        
        echo "${file_type}.zh-en.${lang} complete"
    done
done

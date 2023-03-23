#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# PRETRAIN_MODEL=/generator/data/raw/model/etri_t5/
TRAINSET_PATH=/workspace/data/klue_dataset/ ## guide dataset

BASE_PATH=/workspace # 


# TRAIN_PATH=$TRAINSET_PATH/train.jsonl
# DEV_PATH=$TRAINSET_PATH/dev.jsonl

TEST_PATH=$TRAINSET_PATH/klue_dev.jsonl

#TEMP_PATH=$BASE_PATH/data/mid/inference_metahuman/infer.jsonl
TRAIN_OUTPUT_PATH=$BASE_PATH/model_results/klue_v1/

# mkdir $TRAIN_OUTPUT_PATH

pushd $BASE_PATH/code


# INPUT : 학습완료모델, 테스트데이터
# OUTPUT : 추론결과
python infer.py \
--trained_model $TRAIN_OUTPUT_PATH \
--test_fpath $TEST_PATH \
--output_fpath $TRAINSET_PATH/infer.jsonl \
\
--process_model seq2seq_jsonl \
--model_name_or_path '' \
--data_dir '' \
--file_type '' \
--output_dir '' \
--data_name klue_ner \
--overwrite_output_dir \
--per_device_eval_batch_size 8 \
--predict_with_generate


popd


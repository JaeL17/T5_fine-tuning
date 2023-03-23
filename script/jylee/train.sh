#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PRETRAIN_MODEL=/workspace/data/raw/model/etri_t5/
TRAINSET_PATH=/workspace/data/answer_cls/ 

TRAIN_PATH=$TRAINSET_PATH/train.jsonl
DEV_PATH=$TRAINSET_PATH/dev.jsonl
TEST_PATH=$TRAINSET_PATH/dev.jsonl

TRAIN_OUTPUT_PATH=/workspace/model_results/answer_cls_v1

mkdir $TRAIN_OUTPUT_PATH

pushd /workspace/code

# INPUT : 언어모델, 학습데이터, 개발데이터, 테스트데이터
# OUTPUT : 학습완료모델, 체크포인트, 학습로그

python train.py \
--pretrained_model $PRETRAIN_MODEL \
--train_fpath $TRAIN_PATH \
--dev_fpath $DEV_PATH \
--output_dpath $TRAIN_OUTPUT_PATH \
--do_train \
--model_name_or_path '' \
--process_model seq2seq_jsonl \
--data_name klue_ner \
--data_dir '' \
--file_type '' \
--output_dir '' \
--overwrite_output_dir \
--save_steps 2000 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_train_epochs 3.0 \
--do_predict \
--predict_with_generate

popd


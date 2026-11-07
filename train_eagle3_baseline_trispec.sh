#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train eagle3 baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train/main_eagle3.py \
--basepath Qwen/Qwen3-32B \
--datapath train_data_path \
--savedir save_ckpt_path/eagle3_baseline \
--configpath train/Qwen3-32B_config.json \
--deepspeed_config train/ds_config_bf16.json \
--max_len 2048
# --loss_bf16

# train trispec from scratch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train/main_eagle3_ternary.py \
--basepath Qwen/Qwen3-32B \
--datapath train_data_path \
--savedir save_ckpt_path/eagle3_trispec \
--configpath train/Qwen3-32B_config.json \
--deepspeed_config train/ds_config_bf16.json \
--max_len 2048
# --loss_bf16

# train trispec only for adapter
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train/main_eagle3_adapter.py \
--basepath Qwen/Qwen3-32B \
--datapath train_data_path \
--savedir save_ckpt_path/eagle3_adapter \
--configpath train/Qwen3-32B_config.json \
--deepspeed_config train/ds_config_bf16.json \
--ckpt_path backbone_ckpt_path \
--max_len 2048
# --loss_bf16

#!/usr/bin/env bash
set -e

# train_baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --mixed_precision=bf16 train.main_hass_qwen3 \
    --basepath Qwen/Qwen3-32B \
    --tmpdir train_data_path \
    --cpdir save_ckpt_path/hass_baseline \
    --configpath train/Qwen3-32B_config.json \
    --epoch 40 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3 \
    --data_num 68000 \
    --v_w 1.0

# train_trispec_qwen2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --mixed_precision=bf16 train.main_ternary_draft \
    --basepath deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tmpdir train_data_path_qwen2 \
    --cpdir save_ckpt_path/hass_trispec_qwen2 \
    --configpath train/Distill-r1-32B_config.json \
    --epoch 40 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3 \
    --data_num 68000 \
    --v_w 1.0

# train_trispec_qwen3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --mixed_precision=bf16 train.main_ternary_draft_qwen3 \
    --basepath Qwen/Qwen3-32B \
    --tmpdir train_data_path \
    --cpdir save_ckpt_path/hass_trispec_qwen3 \
    --configpath train/Qwen3-32B_config.json \
    --epoch 40 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3 \
    --data_num 68000 \
    --v_w 1.0

# train_trispec_qwen3_adapter
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --mixed_precision=bf16 train.main_hass_qwen3_adapter \
    --basepath Qwen/Qwen3-32B \
    --tmpdir train_data_path \
    --cpdir save_ckpt_path/hass_adapter \
    --configpath train/Qwen3-32B_config.json \
    --ckpt_path backbone_ckpt_path \
    --epoch 40 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3 \
    --data_num 68000 \
    --v_w 1.0

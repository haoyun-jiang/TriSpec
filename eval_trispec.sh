#!/usr/bin/env bash
set -e

# math benchmark
# Ternary-HASS-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_math \
--ea-model-path eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--model-id trispec-hass \
--bench-name math500 \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192

# Ternary-EAGLE3-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_math \
--ea-model-path eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--model-id trispec-eagle3 \
--bench-name math500 \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192 \
--eagle3

# Ternary-HASS-Adapter-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_math \
--ea-model-path all_eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--model-id trispec-hass-adapter \
--bench-name math500 \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192

# Ternary-EAGLE3-Adapter-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_math \
--ea-model-path all_eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--ori-ckpt-path backbone_ckpt_path \
--model-id trispec-eagle3-adapter \
--bench-name math500 \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192 \
--eagle3

# code benchmark
# Ternary-HASS-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_code \
--ea-model-path eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--model-id trispec-hass \
--bench-name humaneval \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192

# Ternary-EAGLE3-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_code \
--ea-model-path eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--model-id trispec-eagle3 \
--bench-name humaneval \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192 \
--eagle3

# Ternary-HASS-Adapter-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_code \
--ea-model-path all_eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--model-id trispec-hass-adapter \
--bench-name humaneval \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192

# Ternary-EAGLE3-Adapter-margin
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_ternary_code \
--ea-model-path all_eagle_ckpt_path \
--ass-model-path Qwen/Qwen3-1.7B \
--base-model-path Qwen/Qwen3-32B \
--ori-ckpt-path backbone_ckpt_path \
--model-id trispec-eagle3-adapter \
--bench-name humaneval \
--temperature 0 \
--diff-threshold 0.5 \
--seed 42 \
--max-new-token 8192 \
--eagle3

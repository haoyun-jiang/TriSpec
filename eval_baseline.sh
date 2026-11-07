#!/usr/bin/env bash
set -e

# math_benchmark
# naive
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_baseline_answer_qwen3_math \
--ea-model-path eagle_ckpt_path \
--base-model-path Qwen/Qwen3-32B \
--model-id naive \
--bench-name math500 \
--temperature 0 \
--seed 42 \
--max-new-token 8192

# HASS
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_math \
--ea-model-path eagle_ckpt_path \
--base-model-path Qwen/Qwen3-32B \
--model-id hass \
--bench-name math500 \
--temperature 0 \
--seed 42 \
--max-new-token 8192

# EAGLE3
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_math \
--ea-model-path eagle_ckpt_path \
--base-model-path Qwen/Qwen3-32B \
--model-id eagle3 \
--bench-name math500 \
--temperature 0 \
--seed 42 \
--max-new-token 8192 \
--eagle3

# code_benchmark
# naive
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_baseline_answer_qwen3_code \
--ea-model-path eagle_ckpt_path \
--base-model-path Qwen/Qwen3-32B \
--model-id naive \
--bench-name humaneval \
--temperature 0 \
--seed 42 \
--max-new-token 8192

# HASS
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_code \
--ea-model-path eagle_ckpt_path \
--base-model-path Qwen/Qwen3-32B \
--model-id hass \
--bench-name humaneval \
--temperature 0 \
--seed 42 \
--max-new-token 8192

# EAGLE3
CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_qwen3_code \
--ea-model-path eagle_ckpt_path \
--base-model-path Qwen/Qwen3-32B \
--model-id eagle3 \
--bench-name humaneval \
--temperature 0 \
--seed 42 \
--max-new-token 8192 \
--eagle3

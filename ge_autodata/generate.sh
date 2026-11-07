#!/usr/bin/env bash
set -e

# Aeala/ShareGPT_Vicuna_unfiltered
python -m ge_autodata.generate \
--dataset your/sharegpt/path \
--vllm_url http://127.0.0.1:9001/v1/chat/completions \
--output_dir output/path \
--temperature 0.0 \
--max_tokens 2048 \
--start_index 0 \
--end_index 68000 \
--batch_save 10000 \
--threads 256 \
--timeout 600 \
--mode threadpool

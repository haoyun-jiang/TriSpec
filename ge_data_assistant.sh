#!/usr/bin/env bash
set -e

python -m ge_data.allocation \
--outdir output_path \
--data_path train_data_path \
--model_path Qwen/Qwen3-32B \
--ass_path Qwen/Qwen3-1.7B \
--s 0 \
--e 68000 \
--gpu-groups "0;1;2;3"

#!/usr/bin/env bash
set -e

vllm serve Qwen/Qwen3-32B --host 0.0.0.0 --port 9001 --tensor_parallel_size 8 --gpu_memory_utilization 0.9

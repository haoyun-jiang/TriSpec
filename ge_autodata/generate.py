# -*- coding: utf-8 -*-
import json
import requests
import time
import random
import argparse
import os
import threading
import queue
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import pyarrow.parquet as pq
import os
import glob
import re

class ShareGPTReGenerator:
    """A class for regenerating responses in the ShareGPT dataset."""

    def __init__(self, vllm_url: str, output_dir: str, 
                 temperature: float = 0.7, max_tokens: int = 2048,
                 num_threads: int = 4, request_timeout: int = 60,
                 model: str = "Qwen/Qwen3-32B"):
        """
        Initialize the ShareGPT response regenerator.

        Args:
            vllm_url: The API endpoint URL for the vLLM service.
            output_dir: Directory to store the regenerated dataset.
            temperature: Sampling temperature for text generation.
            max_tokens: Maximum number of tokens to generate per request.
            num_threads: Number of concurrent threads for processing.
            request_timeout: Timeout for API requests in seconds.
        """
        self.vllm_url = vllm_url
        self.model = model
        self.output_dir = output_dir
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self.request_timeout = request_timeout

        # Thread-safe synchronization primitives
        self.result_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.api_rate_limit_lock = threading.Lock()
        self.last_api_call = 0
        self.min_api_interval = 0.1  # Minimum interval between API calls (in seconds)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def load_ultra_dataset(self, directory_path: str) -> List[Dict]:
        """Load dataset stored in Parquet format from the specified directory."""
        parquet_files = glob.glob(os.path.join(directory_path, "*.parquet"))
        result_list = []

        for file_path in sorted(parquet_files):
            table = pq.read_table(file_path)
            records = table.to_pydict()

            num_rows = len(next(iter(records.values())))
            row_list = [
                {key: records[key][i] for key in records} 
                for i in range(num_rows)
            ]
            result_list.extend(row_list)

        return result_list

    def load_sharegpt_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load the ShareGPT dataset.

        Args:
            dataset_path: Path to the dataset file, which should be in JSONL or JSON format.

        Returns:
            The loaded dataset as a list of dictionaries.
        """
        data = []
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        return data

    def call_vllm_api(self, prompt: str) -> str:
        """
        Call the vLLM API to generate a response.

        Args:
            prompt: The input prompt string.

        Returns:
            The generated response text.
        """
        try:
            # Enforce a simple rate-limiting mechanism
            with self.api_rate_limit_lock:
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                if time_since_last_call < self.min_api_interval:
                    time.sleep(self.min_api_interval - time_since_last_call)
                self.last_api_call = time.time()

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": []  # Stop tokens can be added as needed
            }

            response = requests.post(self.vllm_url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            raise RuntimeError("Generation request failed; no empty response was substituted") from e

    def process_conversation(self, conversation: Dict, idx: int) -> Dict:
        """
        Process a single conversation by regenerating assistant responses.

        Args:
            conversation: The original conversation data.
            idx: Index of the conversation in the dataset.

        Returns:
            The updated conversation with regenerated responses.
        """
        conversations_list = conversation.get("conversations", [])

        if not conversations_list:
            if "items" in conversation:
                conversations_list = conversation["items"]
            else:
                conversations_list = conversation.get("messages", [])

        new_conversations = []
        context = ""

        for i, msg in enumerate(conversations_list):
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")

            if role in ["human", "user"]:
                new_conversations.append({
                    "from": "human",
                    "value": content
                })
                context += f"Human: {content}\n\n"

            elif role in ["gpt", "assistant"]:
                prompt = f"{context}Assistant:"
                new_response = self.call_vllm_api(prompt)

                new_conversations.append({
                    "from": "gpt",
                    "value": new_response
                })
                context += f"Assistant: {new_response}\n\n"

        new_data = {
            "id": conversation.get("id", f"regenerated_{idx}"),
            "conversations": new_conversations
        }

        return new_data

    def _regenerate(self, conversations, start_index=0, end_index=None, batch_save=10, parallel=True):
        end_index = len(conversations) if end_index is None else end_index
        if not (0 <= start_index < end_index <= len(conversations)):
            raise ValueError("Expected 0 <= start_index < end_index <= dataset size")
        if batch_save < 1 or self.num_threads < 1:
            raise ValueError("batch_save and threads must be positive")
        if glob.glob(os.path.join(self.output_dir, "regenerated_*.json*")):
            raise FileExistsError("Use a fresh output directory to avoid mixing generation runs")
        results = []
        # Collect in input order; IDs may be arbitrary strings, not numeric indices.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads if parallel else 1) as executor:
            for start in range(start_index, end_index, batch_save):
                end = min(start + batch_save, end_index)
                futures = [executor.submit(self.process_conversation, conversations[i], i)
                           for i in range(start, end)]
                batch = [f.result() for f in tqdm(futures)]  # propagate failures
                self.save_batch(batch, start, end - 1)
                results.extend(batch)
        return results

    def regenerate_responses_parallel(self, conversations, start_index=0, end_index=None, batch_save=10):
        return self._regenerate(conversations, start_index, end_index, batch_save)

    def save_batch(self, data: List[Dict], start_idx: int, end_idx: int):
        """
        Save a batch of processed data to disk.

        Args:
            data: List of conversation data to save.
            start_idx: Starting index of the batch.
            end_idx: Ending index of the batch.
        """
        batch_file = os.path.join(self.output_dir, f"regenerated_{start_idx}_to_{end_idx}.json")
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        batch_jsonl = os.path.join(self.output_dir, f"regenerated_{start_idx}_to_{end_idx}.jsonl")
        with open(batch_jsonl, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Batch saved: indices {start_idx} to {end_idx}, total {len(data)} conversations.")

    def regenerate_responses_with_threadpool(self, conversations, start_index=0, end_index=None, batch_save=10):
        return self._regenerate(conversations, start_index, end_index, batch_save)

    def regenerate_responses(self, conversations, start_index=0, end_index=None, batch_save=10):
        return self._regenerate(conversations, start_index, end_index, batch_save, parallel=False)

    def merge_batches(self):
        """Merge all batch files into a single complete dataset file."""
        all_data = []
        batch_files = [f for f in os.listdir(self.output_dir) if re.fullmatch(r"regenerated_\d+_to_\d+\.json", f)]

        for batch_file in sorted(batch_files, key=lambda x: int(x.split('_')[1])):
            with open(os.path.join(self.output_dir, batch_file), 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_data.extend(batch_data)

        complete_file = os.path.join(self.output_dir, "regenerated_complete.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        complete_jsonl = os.path.join(self.output_dir, "regenerated_complete.jsonl")
        with open(complete_jsonl, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Merging completed. Total conversations: {len(all_data)}")


def main():
    parser = argparse.ArgumentParser(description='Regenerate responses in the ShareGPT dataset using vLLM.')
    parser.add_argument('--model', default='Qwen/Qwen3-32B', help='Served model name')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the ShareGPT dataset file.')
    parser.add_argument('--vllm_url', type=str, required=True, help='API endpoint URL for the vLLM service.')
    parser.add_argument('--output_dir', type=str, default='./regenerated_data', help='Output directory for regenerated data.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for text generation.')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate per request.')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index for processing.')
    parser.add_argument('--end_index', type=int, default=None, help='Ending index (exclusive).')
    parser.add_argument('--batch_save', type=int, default=10, help='Number of conversations to process before saving a batch.')
    parser.add_argument('--threads', type=int, default=4, help='Number of concurrent threads for processing.')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout for API requests in seconds.')
    parser.add_argument('--mode', type=str, default='threadpool', 
                        choices=['single', 'parallel', 'threadpool'],
                        help='Processing mode: single-threaded (single), custom multi-threading (parallel), or thread pool (threadpool).')

    args = parser.parse_args()

    regenerator = ShareGPTReGenerator(
        vllm_url=args.vllm_url,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_threads=args.threads,
        request_timeout=args.timeout,
        model=args.model
    )

    if os.path.isdir(args.dataset):
        dataset = regenerator.load_ultra_dataset(args.dataset)
        print(f"Loading Ultra dataset: {args.dataset}")
    else:
        print(f"Loading ShareGPT dataset: {args.dataset}")
        dataset = regenerator.load_sharegpt_dataset(args.dataset)

    print(f"Dataset loading completed. Total conversations: {len(dataset)}")

    if args.mode == 'single':
        regenerator.regenerate_responses(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )
    elif args.mode == 'parallel':
        regenerator.regenerate_responses_parallel(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )
    else:  # threadpool
        regenerator.regenerate_responses_with_threadpool(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )

    regenerator.merge_batches()


if __name__ == "__main__":
    main()

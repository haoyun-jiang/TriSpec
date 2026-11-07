"""Generate benchmark answers. See the repository README."""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from accelerate.utils import set_seed
set_seed(0)

import time

import re
import textwrap
import shortuuid
from tqdm import tqdm

from model.kv_cache import initialize_past_key_values
from model.utils import *


import random
import torch
import numpy as np

import gzip
import json
import re
from evaluation.mbpp import *
from evaluation.humaneval import *
from evaluation.eval_correctness_mbpp.evaluation import evaluate_functional_correctness_mbpp
from evaluation.eval_correctness_humaneval.evaluation import evaluate_functional_correctness_humaneval


def load_questions(question_file, begin=None, end=None):
    with open(question_file, encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream if line.strip()]
    return questions[begin:end]


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def build_prompt(jobj):
    # use humanevalpack prompt
    signature = re.search(
        rf"def\s+({jobj['entry_point']}.*?):\s*\n", jobj["prompt"]
    ).group(1)
    description = "\n".join(
        [
            line.strip()
            for line in re.search(
                rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", jobj["prompt"], re.DOTALL
            )
            .group(1)
            .split("\n")
        ]
    )
    prompt = (
        f"Write a Python function `{signature}` to solve the following problem:\n"
        f"{description}\n"
        f"{jobj['prompt']}"
    )
    return prompt

def extract_response(text):
    sep = "\n</think>\n\n"
    text = text.split(sep)[-1]
    return text

def extract_code(text, entry_point):
    # 正则表达式匹配代码块
    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


def run_eval(
        base_model_path,
        ea_model_path,
        ass_model_path,
        router_path,
        ori_ckpt_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    if args.bench_name == "humaneval":
        questions = load_questions(question_file, question_begin, question_end)[:100]
    elif args.bench_name == "mbpp":
        questions = list(mbpp_read_data(question_file))[question_begin:question_end][:100]

    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    # shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    if not questions:
        raise ValueError("The selected question range is empty")
    workers = num_gpus_total // num_gpus_per_model
    chunk_size = (len(questions) + workers - 1) // workers
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                ass_model_path,
                router_path,
                ori_ckpt_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        ass_model_path,
        router_path,
        ori_ckpt_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0
    if args.eagle3:
        from model.ea_model_ternary_new_eagle3 import EaModel

        model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        ass_model_path=ass_model_path,
        router_path=router_path,
        ori_ckpt_path=ori_ckpt_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )
    else:
        from model.ea_model_ternary_new import EaModel

        model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        ass_model_path=ass_model_path,
        router_path=router_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        messages = []
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        ass_time_mean = []
        base_time_mean = []
        ass_num_list = []
        base_num_list = []

        # start
        if args.bench_name == "humaneval":
            context, _, trailing_prompt = humaneval_doc_to_text(question)
        elif args.bench_name == "mbpp":
            context, _, trailing_prompt = mbpp_doc_to_text(question)
        prompt = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

        # try:
        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, idx, _, ass_num, base_num, ass_time, base_time = model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=temperature,
            ass_threshold=args.ass_threshold,
            diff_threshold=args.diff_threshold,
            entropy_scale=args.entropy_scale,
            accept_reject_diff=args.accept_reject_diff,
            draft_prob_scale=args.draft_prob_scale,
            log=True,
            is_llama3=True,
            max_new_tokens=max_new_token
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        output_ids = output_ids[0][len(input_ids[0]):]
        # be consistent with the template's stop_token_ids
        stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        if stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        # stop_str = "</s>"
        # if stop_str and output.find(stop_str) > 0:
        #     output = output[: output.find(stop_str)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        response = extract_response(output)

        if args.bench_name == "humaneval":
            model_answer = extract_code(response, question)
        elif args.bench_name == "mbpp":
            model_answer = mbpp_extract_answer(response, question['entry_point'])

        # model_answer = extract_code(response, question["entry_point"])
        
        turns.append(output)
        idxs.append(int(idx))
        new_tokens.append(int(new_token))
        wall_time.append(total_time)
        ass_num_list.append(ass_num)
        base_num_list.append(base_num)
        ass_time_mean.append(sum(ass_time)/len(ass_time))
        base_time_mean.append(sum(base_time)/len(base_time))
        messages.append({
            "role": "assistant",
            "content": output
        })
    print('Warmup done')

    # questions=questions[:100]
    question_id = 0
    raw_outputs= []
    for question in tqdm(questions[:100]):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = []
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            ass_time_mean = []
            base_time_mean = []
            accept_length_lists = []
            ass_num_list = []
            base_num_list = []
            
            if args.bench_name == "humaneval":
                context, _, trailing_prompt = humaneval_doc_to_text(question)
            elif args.bench_name == "mbpp":
                context, _, trailing_prompt = mbpp_doc_to_text(question)
                
            prompt = tokenizer.apply_chat_template(
                context,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, accept_length_list,ass_num, base_num, ass_time, base_time = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                ass_threshold=args.ass_threshold,
                diff_threshold=args.diff_threshold,
                entropy_scale=args.entropy_scale,
                accept_reject_diff=args.accept_reject_diff,
                draft_prob_scale=args.draft_prob_scale,
                log=True,
                is_llama3=True,
                max_new_tokens=max_new_token
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            # stop_str = "</s>"
            # if stop_str and output.find(stop_str) > 0:
            #     output = output[: output.find(stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
            response = extract_response(output)

            if args.bench_name == "humaneval":
                model_answer = extract_code(response, question["entry_point"])
            elif args.bench_name == "mbpp":
                model_answer = mbpp_extract_answer(response, question['entry_point'])

            # response = extract_response(output)
            # model_answer = extract_code(response, question["entry_point"])

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            ass_num_list.append(ass_num)
            base_num_list.append(base_num)
            ass_time_mean.append(sum(ass_time)/len(ass_time))
            base_time_mean.append(sum(base_time)/len(base_time))
            accept_length_lists += accept_length_list
            messages.append({
                "role": "assistant",
                "content": output
            })
        # torch.cuda.empty_cache()
        choices.append({"index": i, "turns": turns, "pred_answer":model_answer, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time, "ass_time": ass_time_mean, "base_time": base_time_mean,  "ass_num": ass_num_list, "base_num": base_num_list, "accept_length": accept_length_lists})

        # Dump answers
        os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(answer_file))), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question_id,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "task_id": question["task_id"],
                "completion": model_answer
            }
            fout.write(json.dumps(ans_json) + "\n")
            raw_outputs.append(ans_json)
        question_id += 1

    if args.bench_name == 'humaneval':
        problem_file = question_file
        final_metrics = evaluate_functional_correctness_humaneval(answer_file,problem_file=problem_file)
    elif args.bench_name == 'mbpp':
        problem_file = question_file
        final_metrics = evaluate_functional_correctness_mbpp(answer_file,problem_file=problem_file,is_mbpp=True)
    
    print("\n=== Final Metrics ===")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))

    metrics_record = {
        "question_id":100000,
        "type": "final_metrics",          # 标识这是一条指标汇总
        "answer_id": shortuuid.uuid(),    # 可复用你已有的 id 生成方式
        "model_id": model_id,
        "bench_name": args.bench_name,
        "tstamp": time.time(),
        "metrics": final_metrics          # 直接塞进来
    }
    with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
        fout.write(json.dumps(metrics_record, ensure_ascii=False) + "\n")

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="Qwen/Qwen3-32B",
                        help="Target model ID or directory")
    parser.add_argument("--ass-model-path", type=str, default="Qwen/Qwen3-1.7B",
                        help="1")
    parser.add_argument("--router-path", type=str, default=None, help="1")
    parser.add_argument("--ori-ckpt-path", type=str, default=None, help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Legacy option; quantized loading is not implemented"
    )
    parser.add_argument("--model-id", type=str, default="llama3-8B_1b-ass-ternary")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="humaneval",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--ass-threshold",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--entropy-scale",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--accept-reject-diff",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--draft-prob-scale",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--eagle3", action="store_true", help="Use eagle3"
    )

    args = parser.parse_args()

    if args.load_in_8bit:
        parser.error("Quantized loading is not implemented by this evaluator")
    setup_seed(args.seed)

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"outputs/{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.ass_model_path,
        args.router_path,
        args.ori_ckpt_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)

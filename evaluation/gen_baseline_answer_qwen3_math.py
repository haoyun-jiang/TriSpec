"""Generate benchmark answers. See the repository README."""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from accelerate.utils import set_seed
set_seed(0)

import time

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

from evaluation.examples import get_examples
from evaluation.parser import *
from evaluation.grader import *

N_SHOT={
    "gsm8k": 8,
    "math500": 0,
    "aime24": 0,
    "gaokao2023en": 0,
    "olympiad": 0,
    "mmlu_stem": 0
}


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def load_jsonl(
    file_path,
    instruction="instruction",
    solution="solution",
    output="output",
    category="category",
    choices="choices",
    is_gzip=False,
):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                solution=item[solution] if solution in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
                choices=item[choices] if choices in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

EXAMPLES = get_examples()

def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23", "aime25","math500","gaokao2023en","olympiad"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]

PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
}

def construct_prompt(example, data_name, num_shots):
    prompt_type = "cot"
    demos = load_prompt(data_name, prompt_type, num_shots)

    prompt_temp = PROMPT_TEMPLATES[prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example)
    if num_shots > 0:
        full_prompt = demo_prompt + splitter + context
    else:
        full_prompt = context

    return full_prompt.strip(" ")  # important!

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    # if answer_flag:
    #     # Pick first answer with flag
    #     pred = preds[1]
    # else:
    #     # Pick last number without flag
    #     pred = preds[-1]
    pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

# def clean_answer_math(model_pred):
#     model_pred = model_pred.lower()
#     preds = model_pred.split(ANSWER_TRIGGER.lower())
#     answer_flag = True if len(preds) > 1 else False
#     if answer_flag:
#         # Pick first answer with flag
#         pred = preds[1]
#     else:
#         # Pick last number without flag
#         pred = preds[-1]

#     pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
#     return pred, answer_flag



def run_eval(
        base_model_path,
        ea_model_path,
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
    questions = load_jsonl(question_file, instruction="question",solution='solution', output="answer", choices="choices")[question_begin:question_end]
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
        from model.ea_model_eagle3 import EaModel
    else:
        from model.ea_model import EaModel
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=args.threshold,
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
        is_cor_list = []

        # start
        input_context = question["instruction"]
        if args.bench_name == "mmlu_stem":
            options = question["choices"]
            assert len(options) == 4
            for i, (label, option) in enumerate(zip("ABCD", options)):
                options[i] = f"({label}) {str(option).strip()}"
            options = " ".join(options)
            # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
            input_context = f"{input_context.strip()}\nAnswer Choices: {options}"

        gt_answer = question["output"]
        if args.bench_name == "gsm8k":
            gt_answer = gt_answer
        elif args.bench_name in ["math500"]:
            gt_cot = question["solution"]
            gt_answer = extract_answer(gt_cot, args.bench_name)
        elif args.bench_name in ["olympiad"]:
            gt_answer = gt_answer[0].strip("$")
        elif args.bench_name in ["aime24"]:
            gt_answer = gt_answer
        elif args.bench_name in ["gaokao2023en"]:
            gt_answer = gt_answer.replace("$", "").strip()
        elif args.bench_name == "mmlu_stem":
            abcd = "ABCD"
            gt_answer = abcd[gt_answer]
        else:
            print('please give a valid bench name')
        if args.bench_name not in ["gsm8k"]:
            gt_answer = strip_string(gt_answer)
        qs = construct_prompt(input_context, args.bench_name, N_SHOT[args.bench_name])
        messages.append({
            "role": "user",
            "content": qs
        })
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

        # try:
        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, idx = model.naivegenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=temperature,
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

        if args.bench_name == "gsm8k":
            model_answer = clean_answer(output)
            is_cor = is_correct(model_answer, gt_answer)
        else:
            prediction = extract_answer(output, args.bench_name)
            model_answer = strip_string(prediction)
            is_cor = math_equal(model_answer,gt_answer)
        
        turns.append(output)
        idxs.append(int(idx))
        new_tokens.append(int(new_token))
        wall_time.append(total_time)
        is_cor_list.append(is_cor)
        messages.append({
            "role": "assistant",
            "content": output
        })
    print('Warmup done')

    question_id = 0
    for question in tqdm(questions[:100]):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = []
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            accept_length_lists = []
            is_cor_list = []
            
            input_context = question["instruction"]
            if args.bench_name == "mmlu_stem":
                options = question["choices"]
                assert len(options) == 4
                for i, (label, option) in enumerate(zip("ABCD", options)):
                    options[i] = f"({label}) {str(option).strip()}"
                options = " ".join(options)
                # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
                input_context = f"{input_context.strip()}\nAnswer Choices: {options}"

            gt_answer = question["output"]
            if args.bench_name == "gsm8k":
                gt_answer = gt_answer
            elif args.bench_name in ["math500"]:
                gt_cot = question["solution"]
                gt_answer = extract_answer(gt_cot, args.bench_name)
            elif args.bench_name in ["olympiad"]:
                gt_answer = gt_answer[0].strip("$")
            elif args.bench_name in ["aime24"]:
                gt_answer = gt_answer
            elif args.bench_name in ["gaokao2023en"]:
                gt_answer = gt_answer.replace("$", "").strip()
            elif args.bench_name == "mmlu_stem":
                abcd = "ABCD"
                gt_answer = abcd[gt_answer]
            else:
                print('please give a valid bench name')
            if args.bench_name not in ["gsm8k"]:
                gt_answer = strip_string(gt_answer)
            qs = construct_prompt(input_context, args.bench_name, N_SHOT[args.bench_name])
            messages.append({
                "role": "user",
                "content": qs
            })
                
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx = model.naivegenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
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

            if args.bench_name == "gsm8k":
                model_answer = clean_answer(output)
                is_cor = is_correct(model_answer, gt_answer)
            else:
                prediction = extract_answer(output, args.bench_name)
                model_answer = strip_string(prediction)
                is_cor = math_equal(model_answer,gt_answer)

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            is_cor_list.append(is_cor)
            messages.append({
                "role": "assistant",
                "content": output
            })
            # torch.cuda.empty_cache()
        choices.append({"index": i, "turns": turns, "pred_answer":model_answer, "gt_answer": gt_answer, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time, "accept_length": accept_length_lists, "is_cor": is_cor_list})

        # Dump answers
        os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(answer_file))), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question_id,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
        question_id += 1


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
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="Qwen/Qwen3-32B",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Legacy option; quantized loading is not implemented"
    )
    parser.add_argument("--model-id", type=str, default="llama38b2_40")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="math500",
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
        default=8,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
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
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument("--eagle3", action="store_true", help="Load an EAGLE-3 draft checkpoint")
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

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, required=True)
parser.add_argument('--trainpath', type=str, default="0") 
parser.add_argument('--datapath', nargs='+', type=str, required=True)
parser.add_argument('--savedir', type=str, required=True)
parser.add_argument('--configpath', type=str, required=True)
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument("--max_len", type=int, default=8192)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--loss_bf16', action='store_true')
parser.add_argument('--use_fc_c', action='store_true')

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import shutil
from huggingface_hub import snapshot_download

# Training reads weight shards directly, so resolve a model ID to its local cache.
model_dir = Path(args.basepath).expanduser()
if model_dir.is_dir():
    args.basepath = str(model_dir.resolve())
elif model_dir.is_absolute() or args.basepath.startswith((".", "~")):
    raise FileNotFoundError(f"Model directory does not exist: {args.basepath}")
else:
    args.basepath = snapshot_download(
        args.basepath, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.tiktoken"]
    )

import json
import re
deepspeed_config = args.deepspeed_config

with open(deepspeed_config) as f:
    ds_config = json.load(f)

train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": args.epoch,
    "num_workers": 8,
    "max_len": args.max_len,
    "config_path": args.configpath,
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

import sys

from model.cnets_eagle3 import padding
torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from model.cnets_eagle3 import Model
from model.configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup




def list_files(paths):
    datapath = []
    for path in paths:
        for root, directories, files in os.walk(path):
            for file in files:
                if file.endswith(".ckpt"):
                    file_path = os.path.join(root, file)
                    datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data



class CustomDataset(Dataset):
    def __init__(self, datapath, hidden_size, transform=None, disable_loss_mask=False):
        self.data = datapath
        self.hidden_size = hidden_size
        self.transform = transform
        self.disable_loss_mask = disable_loss_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index],weights_only=False)
        new_data = {}
        hidden_state_dtype = data['hidden_state_target'].dtype
        hidden_state = data['hidden_state_target'][:train_config["max_len"]][None, :]
        # remove [None,:]
        input_ids = data['input_ids'].reshape(1, -1)[:,:train_config["max_len"]]
        loss_mask = data["loss_mask"].reshape(1, -1)

        if self.disable_loss_mask:
            loss_mask = torch.ones_like(loss_mask)

        loss_mask = loss_mask[:, :train_config["max_len"]]
        length = hidden_state.shape[1]

        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :self.hidden_size]
        zeropadding = torch.zeros(1, 1, target.shape[2],dtype=hidden_state_dtype)
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        # 3 layers in eagle3
        new_data["hidden_state_big"] = hidden_state[:,:,self.hidden_size:4*self.hidden_size]
        new_data["input_ids"] = input_ids_target


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        # padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


datapath = list_files(args.datapath)
print("data num: ", len(datapath))


traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

config = EConfig.from_pretrained(train_config["config_path"])
traindataset = CustomDataset(traindatapath, hidden_size=config.hidden_size)
testdataset = CustomDataset(testdatapath, hidden_size=config.hidden_size)


tokenizer = AutoTokenizer.from_pretrained(args.basepath)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, path=args.basepath, load_emb=True, load_head=True,use_fc_c=args.use_fc_c)

model.scandata(args.trainpath, args.basepath)



criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

print("global rank: ", global_rank)


if global_rank == 0:
    import swanlab

    swanlab.init(project=os.environ.get("SWANLAB_PROJECT", "trispec"),
                 experiment_name=Path(args.savedir).name,
                 mode=os.environ.get("TRISPEC_TRACKING", "disabled"))

os.makedirs(args.savedir, exist_ok=True)

sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
                         collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1

checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)

if checkpoint_path:
    print(f"load from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)


for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]


    for batch_idx, data in enumerate(tqdm(train_loader)):

        model.zero_grad()
        plosses, vlosses, acces = model_engine(
                                                hidden_states=data["hidden_states"].to(rank),
                                                input_ids=data["input_ids"].to(rank),
                                                target=data["target"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"].to(rank),
                                                loss_bf16=args.loss_bf16,
                                               )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
 
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)


        model_engine.step()

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            swanlab.log(logdict)
            
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            swanlab.log({f"train/epochacc_{i}": acc_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            swanlab.log({f"train/epochploss_{i}": loss_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            plosses, vlosses, acces = model_engine(
                                                hidden_states=data["hidden_states"].to(rank),
                                                input_ids=data["input_ids"].to(rank),
                                                target=data["target"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"].to(rank),
                                               loss_bf16=args.loss_bf16,
                                                   )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            swanlab.log({f"test/epochacc_{i}": acc_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            swanlab.log({f"test/epochploss_{i}": loss_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")


    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if global_rank == 0:
        shutil.copyfile(args.configpath, Path(f"{args.savedir}/state_{epoch}") / "config.json")

    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")

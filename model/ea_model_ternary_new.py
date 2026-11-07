import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig, LlamaForCausalLM, Qwen3ForCausalLM


from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLMASS
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLMASS
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLMASS
from .r2r_router import HiddenStatesTokenLMHeadLogitsClassifier as Router
from .utils_ternary_new import *
from .kv_cache import initialize_past_key_values

from .cnets_ternary import Model as Model
from .cnets_ternary_qwen3 import Model as ModelQwen3
from .configs import EConfig

from safetensors.torch import load_file





class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ass_model,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            router_state_dict
    ):

        super().__init__()
        self.base_model = base_model
        self.ass_model = ass_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        config = EConfig.from_pretrained(ea_model_path)
        Type=AutoConfig.from_pretrained(base_model_name_or_path).architectures[0]
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        if Type=='LlamaForCausalLM' or Type == 'Qwen2ForCausalLM':
            self.ea_layer = Model(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)
        elif Type == 'Qwen3ForCausalLM':
            self.ea_layer = ModelQwen3(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)
        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

        if router_state_dict is not None:
            self.router = Router(hidden_states_size=router_state_dict['init_args']['hidden_states_size'], logits_size=router_state_dict['init_args']['logits_size'],hidden_dims=router_state_dict['init_args']['hidden_dims'],
                dropout_rate=0, dtype=self.base_model.dtype,pretrained_model=ass_model,freeze_lm_head=True).to(device)
            self.router.load_state_dict(router_state_dict['state_dict'])
        else:
            self.router = None
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            ass_model_path=None,
            router_path = None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = LlamaForCausalLM.from_pretrained(base_model_path,**kwargs)
            ass_model = KVLlamaForCausalLMASS.from_pretrained(
                ass_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLMASS.from_pretrained(base_model_path,**kwargs)
            ass_model = KVQwen3ForCausalLMASS.from_pretrained(
                ass_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLMASS.from_pretrained(base_model_path,**kwargs)
            ass_model = KVQwen2ForCausalLMASS.from_pretrained(
                ass_model_path, **kwargs
            )
            
        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        
        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        
        if router_path is not None:
            # load_router_path = os.path.join(router_path, "model.safetensors")
            router_state_dict = torch.load(router_path,map_location=base_model.device,weights_only=False)
        else:
            router_state_dict = None

        model = cls(
            base_model,
            base_model_path,
            ass_model,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            router_state_dict
        )



        if total_token==-1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]
            times=[]

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token=cans[times.index(min(times))]
            model.ea_layer.total_tokens=total_token-1




        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.ass_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.ass_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    def forward_base(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            ass_threshold=0.7,
            diff_threshold=0.0,
            entropy_scale=0.0,
            accept_reject_diff=0.0,
            draft_prob_scale=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=1024,
            max_length=40960,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        padding_probs=(torch.zeros(1,dtype=torch.float)).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.ass_model)
            self.past_key_values = past_key_values 
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        if hasattr(self, "base_past_key_values"):
            base_past_key_values = self.base_past_key_values
            base_past_key_values_data = self.base_past_key_values_data
            base_current_length_data = self.base_current_length_data
            # Reset the past key and value states
            base_current_length_data.zero_()
        else:
            (
                base_past_key_values,
                base_past_key_values_data,
                base_current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.base_past_key_values = base_past_key_values 
            self.base_past_key_values_data = base_past_key_values_data
            self.base_current_length_data = base_current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self.ass_model)
        reset_tree_mode(self.base_model)
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token, draft_probs = initialize_tree_ternary(
            input_ids, self, past_key_values, base_past_key_values, logits_processor
        )
        # draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
        #     input_ids, self, past_key_values, logits_processor
        # )

        new_token = 0
        accept_length_list = []
        
        ass_num = 0
        base_num = 0
        ass_time = []
        base_time = []
        for idx in range(max_length):
            #with Timer("all"):
            self.ass_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            start_time = time.time()
            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1) 
            candidates=draft_tokens[0,retrieve_indices]
            draft_probs = torch.cat((draft_probs,padding_probs),dim=0) 
            draft_probs = draft_probs[retrieve_indices]
            candidates_hidden_states=hidden_state_new[0,retrieve_indices]
            best_candidate, accept_length_ass, sample_p, use_ass = evaluate_posterior(
                logits, candidates, draft_probs, logits_processor, ass_threshold, diff_threshold,entropy_scale, accept_reject_diff, draft_prob_scale, candidates_hidden_states, self.router
            )

            # accept_token_length = current_length_data[0] - base_current_length_data[0] - draft_tokens.shape[1]+1
            # if accept_token_length > 64 and use_ass:
            #     use_ass = False
            #     accept_length_ass = torch.tensor(0)
            
            if use_ass:
                accept_length = accept_length_ass
                best_candidate = best_candidate[0]
            else:
                # retrieve_indices = retrieve_indices[best_candidate,accept_length_ass]
                draft_tokens = draft_tokens[:,:-1]
                accept_token_length = current_length_data[0] - base_current_length_data[0] - draft_tokens.shape[1]
                if accept_token_length > 512:
                    print("accept_token_length is too large")
                    break
                if accept_token_length == 0:
                    retrieve_indices = retrieve_indices.to(input_ids.device)
                    base_tree_mask = tree_mask
                    base_draft_tokens = draft_tokens
                    base_tree_position_ids = tree_position_ids
                    base_input_ids = input_ids
                    base_retrieve_indices = retrieve_indices[best_candidate]
                else:
                    retrieve_indices = retrieve_indices.to(input_ids.device)
                    base_tree_mask = create_base_tree_mask(tree_mask,accept_token_length)
                    base_input_ids = input_ids[:,:-accept_token_length]
                    base_draft_tokens = torch.cat([input_ids[:,-accept_token_length:],draft_tokens],dim=-1)
                    base_tree_position_ids = torch.cat([torch.arange(accept_token_length).to(input_ids.device),tree_position_ids+accept_token_length],dim=0)
                    base_retrieve_indices = retrieve_indices[best_candidate]
                    base_retrieve_indices[base_retrieve_indices != -1] += accept_token_length
                self.base_model.model.tree_mask = base_tree_mask
                logits_base, hidden_state_new_base, outputs_base = tree_decoding_base(
                    self,
                    base_draft_tokens,
                    base_past_key_values,
                    base_tree_position_ids,
                    base_input_ids,
                    base_retrieve_indices,
                )
                
                base_verify_num = retrieve_indices.shape[1] - accept_length_ass.item()
                base_draft_tokens=torch.cat((base_draft_tokens,padding),dim=1) 
                candidates_base=base_draft_tokens[0,base_retrieve_indices]
                if base_verify_num == 1:
                    best_candidate_base, accept_length_base, sample_p_base = 0, 0, logits_base[0, -1] 
                else:
                    best_candidate_base, accept_length_base, sample_p_base = evaluate_posterior_base(
                        logits_base, candidates_base, logits_processor, base_verify_num
                    )
                accept_length = accept_length_ass + accept_length_base
            end_time = time.time()
            total_time = end_time - start_time
            if use_ass:
                ass_time.append(total_time)
            else:
                base_time.append(total_time)
                

            try:
                accept_length_list.append(accept_length.item())
            except:
                accept_length_list.append(accept_length)


            # print(accept_length)
            #with Timer("update_inference_inputs"):
            if use_ass:
                input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token, draft_probs = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state_new,
                    sample_p,
                )
                ass_num += 1
            else:
                input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token, draft_probs = update_inference_inputs_base(
                    input_ids,
                    candidates_base,
                    best_candidate_base,
                    accept_token_length,
                    accept_length_ass,
                    accept_length_base,
                    base_retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    base_past_key_values_data,
                    current_length_data,
                    base_current_length_data,
                    self,
                    hidden_state_new_base,
                    sample_p_base,
                )
                base_num += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, accept_length_list, ass_num, base_num, ass_time, base_time


    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token+=1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.ass_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            #with Timer("all"):
            self.ass_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break


    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>") # <|im_end|>
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0


        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids



            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break




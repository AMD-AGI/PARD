import sys
import click
import time
import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, AutoConfig
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSSSSS}</green> | {level} | {message}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
np.random.seed(42)
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._dynamo.config.cache_size_limit = 128


@click.command()
@click.option('-k', '--draft_k', default=12)
@click.option('--tokens', default=512)
@click.option('-d', '--draft', default='qwen_0.5b_pard')
@click.option('-t', '--target', default='qwen_2.5_7b')
@click.option('-b', '--benchmark', default='humaneval')
@click.option('-ms', '--model_serie', default=None)
@click.option('--para', is_flag=True)
@click.option('--nc', is_flag=False)
@click.option('--maxtune', is_flag=True)
@click.option('--max_cache_len', default=None)
def main(draft_k, tokens, draft, target, benchmark, para, nc, maxtune, max_cache_len, model_serie):
    infer = pard(draft_k, tokens, draft, target, benchmark, para, nc, maxtune, max_cache_len, model_serie)
    infer.eval()


class PardInfer():
    def __init__(self, draft_k, tokens, draft, target, benchmark, para, nc=False, maxtune=False, max_cache_len=None, model_serie=None):
        self.draft_k = draft_k
        self.tokens = tokens
        self.checkpoint_draft = draft
        self.checkpoint_target = target
        self.torch_dtype = torch.bfloat16
        self.device = torch.device("cuda")
        self.benchmark = benchmark
        self.para = para
        self.nc = nc
        self.maxtune = maxtune
        self.max_cache_len = max_cache_len if max_cache_len is not None else self.tokens
        self.warm_up_num = 8
        self.log = True
        if self.para:
            draft_config = AutoConfig.from_pretrained(self.checkpoint_draft)
            self.pard_token = [draft_config.pard_token for i in range(32)]
        if model_serie is not None:
            self.model_serie = model_serie
        else:
            # auto model serie
            if 'llama3' in target or 'Llama-3' in target:
                self.model_serie = 'llama3'
            elif 'r1' in target or 'R1' in target:
                self.model_serie = 'r1'
            elif 'qwen' in target or 'Qwen' in target:
                self.model_serie = 'qwen'

    def get_data(self, data_path):
        return [json.loads(i)['data'] for i in open(data_path).readlines()]

    def model_init(self, path, torch_dtype=torch.bfloat16, max_cache_len=512, device='cuda'):
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype,
            attn_implementation='eager',
        ).to(device)
        cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=model.device,
            dtype=model.dtype
        )
        return model, cache

    def get_model(self, checkpoint_draft, checkpoint_target, model_serie, torch_dtype, max_cache_len, nc, maxtune):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_target)
        self.model, self.target_cache = self.model_init(checkpoint_target, torch_dtype=torch_dtype, max_cache_len=max_cache_len)
        self.model_draft, self.draft_cache = self.model_init(checkpoint_draft, torch_dtype=torch_dtype, max_cache_len=max_cache_len)

        if not nc:
            draft_mode = 'max-autotune'
            target_mode = 'max-autotune' if maxtune else 'reduce-overhead'
            self.model.forward_gen = torch.compile(self.model.forward, mode=target_mode, fullgraph=True, dynamic=False)
            self.model_draft.forward_gen = torch.compile(self.model_draft.forward, mode=draft_mode, fullgraph=True)
        else:
            self.model.forward_gen = self.model.forward
            self.model_draft.forward_gen = self.model_draft.forward

    def get_input(self, prompt, tokenizer, prompt_type):
        if prompt_type != 'r1':
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer([text],add_special_tokens=False,return_tensors='pt').input_ids.cuda()
        return  text, input_ids

    def eval(self):
        test_datas = self.get_data(f'datas/bmk/{self.benchmark}.jsonl')
        self.get_model(self.checkpoint_draft, self.checkpoint_target, self.model_serie, self.torch_dtype, self.max_cache_len, self.nc, self.maxtune)
        self.generate(test_datas[:self.warm_up_num], self.model, self.model_draft, self.target_cache, self.draft_cache, tokenizer=self.tokenizer) ##warm up
        output = self.generate(test_datas, self.model, self.model_draft, self.target_cache, self.draft_cache, tokenizer=self.tokenizer)
        return output

    def speed_logger(self, name=None, log=True):
        torch.cuda.synchronize()
        time_now = time.time()
        if log and self.log:
            logger.info(f'[{name}]: {1 / (time_now - self.time_last)}')
        self.time_last = time_now
        return time_now

    def cache_mask(self, cache, seq_length_start, cache_length):
        if seq_length_start >= cache_length:
            return cache_length
        else:
            return seq_length_start


    def generate(self, test_datas, model, model_draft, target_cache, draft_cache, tokenizer):
        profile = {
            'acc': [],
            'draft': [],
            'target': [],
            'target_prefill': [],
            'accept_length': [],
            'time_cost': [],
        }
        with torch.no_grad():
            for text in test_datas:
                ## init input
                prefill_stage = True
                s = self.speed_logger(log=False)
                target_cache.reset()
                target_cache_length = 0
                draft_cache.reset()
                draft_cache_length = 0
                text_new, token_input = self.get_input(text, tokenizer, prompt_type=self.model_serie)
                all_token = token_input
                target_input = {
                        'input_ids': token_input,
                        'position_ids': None,
                        'past_key_values': target_cache,
                        'cache_position': torch.arange(target_cache_length, target_cache_length + token_input.shape[1], device=model.device),
                        'use_cache': True, 
                        'attention_mask': None,
                        }
                draft_input = {
                        'input_ids': token_input,
                        'position_ids': None,
                        'past_key_values': draft_cache,
                        'cache_position': torch.arange(draft_cache_length, draft_cache_length + token_input.shape[1], device=model.device), 
                        'use_cache': True, 
                        'attention_mask': None,
                        }
                input_token_lenght = token_input.shape[1]
                total_token_lenght = token_input.shape[1]
                while (total_token_lenght - input_token_lenght < self.tokens) and (total_token_lenght < (self.max_cache_len - 32)):
                    # draft_sampling
                    draft_tmp_new_token = all_token[:,:0]
                    for k in range(self.draft_k if not self.para else min(self.draft_k, 1)):
                        if self.para:
                            unused_tokenids = torch.tensor([self.pard_token]).to(draft_input['input_ids'].device)[:,:self.draft_k-1]
                            draft_input['input_ids'] = torch.concat([draft_input['input_ids'], unused_tokenids], axis=1)
                            draft_input['cache_position'] = torch.arange(draft_cache_length, draft_cache_length+draft_input['input_ids'].shape[1], device=model.device)
                        ss = self.speed_logger('init')
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                            draft_output = model_draft.forward_gen(**draft_input,  return_dict=True, output_attentions=False, output_hidden_states=False)
                            draft_cache_length += len(draft_input['cache_position'])
                        ee = self.speed_logger('draft')
                        profile['draft'].append(ee-ss)
                        if self.para:
                            new_token_ids = draft_output.logits[:,-self.draft_k:].argmax(-1)
                        else:
                            new_token_ids = draft_output.logits[:,-1:].argmax(-1)
                        draft_input = {
                            'input_ids': new_token_ids,
                            'position_ids': None,
                            'past_key_values': draft_cache,
                            'cache_position': torch.arange(draft_cache_length, draft_cache_length + new_token_ids.shape[1], device=model.device),
                            'use_cache': True, 
                            'attention_mask': None,
                            }
                        draft_tmp_new_token = torch.concat([draft_tmp_new_token, new_token_ids], axis=1)
                    # target_forward
                    target_input['input_ids'] = torch.concat([target_input['input_ids'], draft_tmp_new_token], axis=1)
                    target_input['cache_position'] = torch.arange(target_cache_length, target_cache_length + target_input['input_ids'].shape[1], device=model.device)
                    ss = self.speed_logger(log=False)
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        if prefill_stage:
                            model_run = model.forward
                        else:
                            model_run = model.forward_gen
                        target_output = model_run(**target_input,  return_dict=True, output_attentions=False, output_hidden_states=False)
                        target_cache_length += len(target_input['cache_position'])
                    ee = self.speed_logger('target')
                    if prefill_stage:
                        s = self.speed_logger(log=False)
                        prefill_stage = False
                    new_token_ids = target_output.logits[:,-draft_tmp_new_token.shape[1] - 1:].argmax(-1)
                    keep_token_ids = []
                    for i in range(draft_tmp_new_token.shape[1]):
                        ori_input_token_lenght = target_input['input_ids'].shape[1] - draft_tmp_new_token.shape[1]
                        if new_token_ids[0][i] == target_input['input_ids'][0][ori_input_token_lenght + i]:
                            profile['acc'].append(1)
                            keep_token_ids.append(new_token_ids[:,i])
                        else:
                            profile['acc'].append(0)
                            break

                    keep_token_ids.append(new_token_ids[:,len(keep_token_ids)])
                    keep_token_ids = torch.concat(keep_token_ids).reshape(1,-1)
                    all_token = torch.concat([all_token, keep_token_ids], axis=1)
                    logger.info(f'[accept token num]: {len(keep_token_ids[0])}')
                    profile['accept_length'].append(len(keep_token_ids[0]))
                    total_token_lenght += keep_token_ids.shape[1]
                    if tokenizer.eos_token_id in keep_token_ids:
                        break
                    # update_input
                    ## target
                    self.speed_logger('target sampling')
                    target_cache_length = self.cache_mask(target_cache, total_token_lenght - 1, target_cache_length)
                    target_input = {
                        'input_ids': keep_token_ids[:,-1:],
                        'position_ids': None,
                        'past_key_values': target_cache,
                        'cache_position': torch.arange(target_cache_length, target_cache_length +  keep_token_ids[:,-1:].shape[1], device=model.device),
                        'use_cache': True, 
                        'attention_mask': None,
                        }
                    self.speed_logger('update target input')
                    ## draft
                    if self.para:
                        draft_input_ids = keep_token_ids
                        remove_kv_num = self.draft_k - 1
                        draft_cache_length = self.cache_mask(draft_cache, draft_cache_length - remove_kv_num, draft_cache_length)
                    else:
                        draft_input_ids = keep_token_ids[:,-2:] if keep_token_ids.shape[1] ==  self.draft_k + 1 else  keep_token_ids[:,-1:]
                        draft_cache_length = self.cache_mask(draft_cache, total_token_lenght - 1, draft_cache_length)
                    draft_input = {
                        'input_ids': draft_input_ids,
                        'position_ids': None,
                        'past_key_values': draft_cache,
                        'cache_position': torch.arange(draft_cache_length, draft_cache_length + draft_input_ids.shape[1], device=model.device),
                        'use_cache': True, 
                        'attention_mask': None,
                        }
                    self.speed_logger('update draft input')
                output = tokenizer.decode(all_token[0])
                e = self.speed_logger('end')
                logger.info(f'\n{"-"*50}\nA:\n{output}\n{"-"*50}\n')
                logger.info(f'acc: {np.array(profile["acc"]).mean()} | accept_length: {np.array(profile["accept_length"]).mean()}')
                generate_token = total_token_lenght - input_token_lenght
                logger.info(f"[speed] | total: {e-s} | token/s: {generate_token / (e-s)}")
                profile['time_cost'].append([e-s, generate_token])
            token_per_sec = np.array(profile['time_cost'])[:,1].sum() / np.array(profile['time_cost'])[:,0].sum()
            logger.info(f"[speed all] | total: {np.array(profile['time_cost'])[:,0].sum()} | token/s: {token_per_sec}")
            return {'acc': np.array(profile['acc']).mean(), 'token_per_sec': token_per_sec, 'accept_length': np.array(profile["accept_length"]).mean()}

if __name__ == '__main__':
    main()

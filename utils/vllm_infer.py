import argparse
import json
import os
from vllm import LLM, SamplingParams
import requests
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt


import numpy as np


os.environ.update({
    "VLLM_USE_V1": "0"
})
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--draft", type=str, default="amd/PARD-Llama-3.2-1B")
    parser.add_argument("--benchmark", type=str, default="humaneval")
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=80)
    parser.add_argument("--num_spec_tokens", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("-t", "--token", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--ar", action='store_true')
    parser.add_argument("-r", "--reasoning", action='store_true')
    parser.add_argument("--disable-warmup", action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    prompts = []
    for line in requests.get(f'https://raw.githubusercontent.com/AMD-AIG-AIMA/PARD/master/datas/bmk/{args.benchmark}.jsonl').text.splitlines():
        if line:
            prompts.append(json.loads(line)['data'])
    prompts = prompts[:args.num_prompts]
    datas = [[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}] for prompt in prompts]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    texts = []
    for data in datas:
        text = tokenizer.apply_chat_template(
            data,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.reasoning,
        )
        texts.append(text)
    batch_input_ids = tokenizer(texts, return_attention_mask=False)['input_ids']
    batch_input_ids = [TokensPrompt(prompt_token_ids=ids) for ids in batch_input_ids]

    llm = LLM(
        model=args.model,
        enable_prefix_caching=False,
        tensor_parallel_size=args.tp,
        max_model_len=8192,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.8,
        speculative_config=None if args.ar else {
            "model": args.draft,
            "num_speculative_tokens": args.num_spec_tokens
            },
        compilation_config={
            "splitting_ops": [],
            "compile_sizes": [],
            "cudagraph_capture_sizes": [
                256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,
                 120,112,104,96,88,80,72,64,56,48,40,34,33,32,3130,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1
            ],
            "max_capture_size": 256
        },
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.token)

    ## warmup
    if not args.disable_warmup:
        print("warmup...")
        outputs = llm.generate(batch_input_ids, sampling_params=sampling_params)

    # inference
    print("inference...")
    outputs = llm.generate(batch_input_ids, sampling_params=sampling_params)

    # speed
    speed = []
    for output in outputs:
        speed.append([len(output.outputs[0].token_ids), (output.metrics.finished_time - output.metrics.first_token_time)])
        print(f"[anwer]:\n {output.outputs[0].text}")

    print(f"\n\n{'='*100}\n\n")
    print(f'[speed]: {np.array(speed)[:,0].sum() / np.array(speed)[:,1].sum()}\n')

    # accepted
    if not args.ar:
        acceptance_counts = [0] * (args.num_spec_tokens + 1)
        for output in outputs:
            for step, count in enumerate(
                    output.metrics.spec_token_acceptance_counts):
                acceptance_counts[step] += count

        print(f"[acceptance length]: {(sum(acceptance_counts) / acceptance_counts[0])}")
    print(f"\n\n{'='*100}\n\n")
    print(args.__dict__)
    print(f"\n\n{'='*100}\n\n")


if __name__ == "__main__":
    main()

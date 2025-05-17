import click
import os
import yaml
import itertools
import torch
import json
from datetime import datetime
import multiprocessing
import gc
from pard.pard_infer import PardInfer

def worker(kwargs, queue):
    infer = PardInfer(**kwargs)
    result = infer.eval()
    kwargs['result'] = result
    kwargs['device'] = torch.cuda.get_device_name(0)
    torch._dynamo.reset()
    del infer
    torch.cuda.empty_cache()
    gc.collect()
    queue.put(kwargs)

@click.command()
@click.option('-c', '--config_path', default='config/eval/eval.yaml')
def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        eval_kwargs_all = yaml.safe_load(f)
    
    out = []
    for eval_kwargs in eval_kwargs_all['eval']:
        keys = list(eval_kwargs.keys())
        values_lists = [eval_kwargs[key] for key in keys]
        
        for combination in itertools.product(*values_lists):
            kwargs = dict(zip(keys, combination))
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=worker, args=(kwargs, q))
            p.start()
            p.join()  
            result = q.get()
            out.append(result)
            print(f'[eval tmp result]: \n{result}\n{"-"*100}\n\n')
    
    print(f'{"="*100}\n\n[eval all result]: \n')
    for i in out:
        print(f'{i} \n\n {"-"*100}')
    
    save_base_path = 'datas/result'
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    save_path = os.path.join(save_base_path, datetime.now().strftime("%Y%m%d_%H%M%S") + '.jsonl')
    with open(save_path, "w", encoding="utf-8") as f:
        for item in out:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()

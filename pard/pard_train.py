import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling
import yaml
from loguru import logger
import click
torch.set_printoptions(linewidth=200)

@click.command()
@click.option('-c', '--config_path')
def main(config_path):
    config = yaml.safe_load(open(config_path))
    config_general = config['general']
    model_name = config_general['model_name']
    cache_dir = config_general.get('cache_dir', 'cache')
    exp_name = config_general.get('exp_name', None)
    exp_name = model_name.split('/')[-1] + '_' + (exp_name if exp_name is not None else os.path.basename(os.getcwd()))
    output_dir = os.path.join(config_general['save_base_path'], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"cp {config_path} {os.path.join(output_dir, 'config.yaml')}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pard_token = config_general.get('unused_tokenids', [-1])[0]
    model.config.spd_type = 'pard'
    
    dataset = load_dataset("json", split="train", data_files=config_general['data_path'])

    logger.info(f'\n[model_path]: \n{model_name}')
    logger.info(f'\n[save_path]: \n{output_dir}')
    
    def formatting_prompts_func(example):
        key_name = 'conversation'
        output_texts = []
        for i in range(len(example[key_name])):
            prompt = example[key_name][i][0]['human']
            answer = example[key_name][i][0]['assistant']
            text = config['data']['prompt_template'][config['data']['prompt_type']].format(prompt=prompt, answer=answer)
            output_texts.append(text)
        return output_texts
    
    sft_config = SFTConfig(
        output_dir=output_dir,
        bf16=True,
        **config['train']
    )

    if sft_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    class warp_data_collator():
        def __init__(self, tokenizer, para_num=4, unused_tokenids=[128020, 128021, 128022, 128023], down_sample_ration=1, down_sample_ration_min=0):
            self.tokenizer = tokenizer
            self.unused_tokenids = unused_tokenids
            self.para_num = para_num
            self.down_sample_ration = down_sample_ration
            self.down_sample_ration_min = down_sample_ration_min
            self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
    
        def __call__(self, data):
            # Para Data
            data = self.data_collator(data)
            unused_tokenids = self.unused_tokenids
            para_num = self.para_num
            single_lenght = data['input_ids'].shape[1]
            tgt_len = single_lenght * para_num
            mask = torch.full((tgt_len, tgt_len), torch.finfo(torch.float32).min)
            mask_cond = torch.arange(mask.size(-1))
            tmp_mask = (mask_cond == (mask_cond).view(mask.size(-1), 1)) 
            for i in range(para_num):
                tmp_mask += (mask_cond == (mask_cond - single_lenght * i - i).view(mask.size(-1), 1))
                tmp_mask += (mask_cond < (mask_cond - i * single_lenght - (i - 1)).view(mask.size(-1), 1)) * (mask_cond < (i+1) * single_lenght).view(-1,1)
            mask.masked_fill_(tmp_mask, 0)
            mask = mask[None, None, :, :]
            bs = data['input_ids'].shape[0]
            new_data = {
                    'input_ids': torch.concat([data['input_ids'], torch.concat([data['input_ids'] * 0 + unused_tokenids[i] for i in range(para_num-1)], axis=1)], axis=1) if para_num > 1 else data['input_ids'],
                    'attention_mask': torch.concat([mask for i in range(bs)], axis=0),
                    'position_ids': torch.concat([torch.arange(single_lenght, dtype=torch.long) for i in range(para_num)]).unsqueeze(0).repeat(bs, 1) ,
                    'labels': torch.concat([torch.concat([data['labels'][:,:i] * 0 - 100, data['labels'][:,i:]], axis=1) for i in range(para_num)], axis=1),
            }
            # COD
            if self.down_sample_ration != 1 and para_num > 1:
                index_mask = torch.zeros(para_num, single_lenght, dtype=torch.bool)
                index_mask[0, :] = 1
                prev_indices = torch.arange(single_lenght)
                for i in range(1, para_num):
                    num_ones = int(single_lenght * max(self.down_sample_ration ** i, self.down_sample_ration_min))
                    if num_ones == 0:
                        break
                    selected_indices = prev_indices[torch.randperm(len(prev_indices))[:num_ones]]
                    index_mask[i, selected_indices] = 1
                    prev_indices = (selected_indices + 1) % single_lenght
                index_mask = index_mask.reshape(-1)

                indices = index_mask.nonzero(as_tuple=True)[0]
            
                filtered_data = {}
                filtered_data["input_ids"] = new_data["input_ids"][:, indices].contiguous()
                filtered_data["position_ids"] = new_data["position_ids"][:, indices].contiguous()
                filtered_data["labels"] = torch.roll(torch.roll(new_data["labels"], shifts=-1, dims=1)[:, indices], shifts=1, dims=1).contiguous()
                filtered_data["attention_mask"] = new_data["attention_mask"][:, :, indices, :][:, :, :, indices].contiguous()
                new_data = filtered_data
            return new_data
    
    data_collator = warp_data_collator(tokenizer, para_num=config_general.get('para_num', 1), unused_tokenids=config_general.get('unused_tokenids', []), down_sample_ration=config_general.get('down_sample_ration', 1), down_sample_ration_min=config_general.get('down_sample_ration_min', 0))
    
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
    )
    
    trainer.train()
    logger.info(f'\n[save_path]: \n{output_dir}')

if __name__ == '__main__':
    main()

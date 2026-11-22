from src.data.data import get_dataframe
from src.utils import select_label_from_response
from src.prompt_modeling import create_evaluation_prompt
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from omegaconf import OmegaConf
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import time
from peft import PeftModel
from tqdm import tqdm
import numpy as np


def get_model_and_tokenizer(model_id: str, quant: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
  
    if quant:
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
        # )
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id
        )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer


class CustomDataset(Dataset):
    def __init__(self, text_data: List[str]):
        self.dataset = text_data

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def evaluate_on_wa(args):
    
    START_TIME = time.strftime("%Y%m%d_%H%M%S")

    cfg = OmegaConf.load(args.config_file)
    
    set_seed(cfg.random_seed)
    
    df_test_wa = get_dataframe(cfg.dataset, cfg.test_subset, cfg.source)
    
    _formating_prompt = partial(
        create_evaluation_prompt,
        task_type=cfg.task,
        instruction_type=cfg.task,
        label_description_type=cfg.prompt.description_level, 
    )
    
    prompts = df_test_wa.apply(_formating_prompt, axis=1)

    eval_dataset = CustomDataset(prompts)

    device = "cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu"

    model_id=cfg.model_name
    model_name = os.path.split(model_id)[1]
    model, tokenizer = get_model_and_tokenizer(model_id, cfg.quantization)
    tokenizer.padding_side = "left"
    # model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype="auto")
    if cfg.lora_adapter is not None:
        splitted_adapter_name = cfg.lora_adapter.split('/')
        assert os.path.split(cfg.model_name)[1] == splitted_adapter_name[-4]
        assert cfg.dataset == splitted_adapter_name[-5]
        assert cfg.task == splitted_adapter_name[-6]
        model = PeftModel.from_pretrained(model, cfg.lora_adapter)
    model = model.to(device)

    def collate_fn_with_padding(batch):
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest"
        )
        return tokenized
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.loader.batch_size,
        collate_fn=collate_fn_with_padding
    )

    responses = []
    prog = tqdm(eval_loader)
    for i, batch in enumerate(prog):

        inputs_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) 

        outputs = model.generate(
            inputs_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=cfg.generation.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, 
            do_sample=False, 
            temperature=None, 
            top_k=None, 
            top_p=None
        )
        response = tokenizer.batch_decode(outputs[:, len(inputs_ids[0]):], skip_special_tokens=True)
        responses.extend(response)

    preds = [select_label_from_response(response, cfg.task) for response in responses]
    # breakpoint()
    result_df = df_test_wa
    result_df["preds"] = preds
    result_df["response"]= responses
    
    if cfg.lora_adapter:
        adapter_name = cfg.lora_adapter.split('/')[-1]
        base_model_name = os.path.split(cfg.model_name)[1]
        model_name = f"{base_model_name}"
    else:
        model_name = os.path.split(cfg.model_name)[1]

    save_dir = (
        "results_new/"
        f"{cfg.task_adaptation}/"
        f"{cfg.task}/"
        f"{model_name}/"
        f"{cfg.dataset}_"
        f"{cfg.prompt.description_level}_"
        f"{cfg.prompt.n_examples}_"
        f"lr{splitted_adapter_name[-3] if splitted_adapter_name else "None"}_"
        f"lora_rank{splitted_adapter_name[-2] if splitted_adapter_name else "None"}_"
        f"{START_TIME}"
    )
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_csv(os.path.join(save_dir, "results.csv"), index=False)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)

    args = parser.parse_args()
    evaluate_on_wa(args)

if __name__ == "__main__":
    main()
from argparse import ArgumentParser
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import Dataset
import time
from src.prompt_modeling import create_train_prompt
from src.data.data import get_dataframe
from functools import partial
from sklearn.model_selection import train_test_split
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
import os
from trl import SFTTrainer

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

def main(args):
    cfg = OmegaConf.load(args.config_file)
    df_wa = get_dataframe(cfg.data.dataset, cfg.data.subset, cfg.data.source)
    _formating_prompt = partial(create_train_prompt, df=df_wa, 
                                task_type=cfg.task,
                                instruction_type=cfg.task,
                                label_description_type=cfg.prompt.description_level,
                                )
    
    form_dataset = df_wa.apply(_formating_prompt, axis=1)
   
    prompts = [sample[0] for sample in form_dataset]
    completions = [sample[1] for sample in form_dataset]
    # breakpoint()
    START_TIME = time.strftime("%Y%m%d_%H%M%S")
    model_id=cfg.model.base_model
    model_name = os.path.split(model_id)[1]
    output_model=f"models/{cfg.task}/{cfg.data.dataset}/{model_name}/{cfg.training.learning_rate}/{cfg.lora.rank}/{START_TIME}/"
    model, tokenizer = get_model_and_tokenizer(model_id, cfg.model.quantization)

    dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})

    peft_config = LoraConfig(
        r=cfg.lora.rank, 
        lora_alpha=cfg.lora.alpha, 
        lora_dropout=cfg.lora.lora_dropout, 
        bias=cfg.lora.bias, 
        task_type="CAUSAL_LM"
    )
    # model = get_peft_model(model, peft_config)
    training_arguments = TrainingArguments(
       output_dir=output_model,
       num_train_epochs=cfg.training.num_train_epochs,
       per_device_train_batch_size=cfg.training.per_device_train_batch_size,
       gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
       optim=cfg.training.optim,
    #    save_steps=cfg.training.save_steps,
       logging_steps=cfg.training.logging_steps,
       learning_rate=cfg.training.learning_rate,
       weight_decay=cfg.training.weight_decay,
       fp16=cfg.training.fp16,
       bf16=cfg.training.bf16,
       max_grad_norm=cfg.training.max_grad_norm,
       max_steps=cfg.training.max_steps,
       warmup_ratio=cfg.training.warmup_ratio,
       group_by_length=cfg.training.group_by_length,
       lr_scheduler_type=cfg.training.lr_scheduler_type,
       save_strategy=cfg.training.save_strategy
   )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
    )

    trainer.train()
    trainer.save_model(output_model)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=False, default="/home/a.scisel/2024/llm2vec/iSTS-mgr/src/configs/finetune_llm.yaml")
    args = parser.parse_args()
    main(args)
    
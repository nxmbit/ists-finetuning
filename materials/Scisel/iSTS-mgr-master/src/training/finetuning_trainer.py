from argparse import ArgumentParser
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import Dataset
import time
from src.prompt_modeling import create_prompt
from src.data.data import get_dataframe
from functools import partial
from sklearn.model_selection import train_test_split
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
import os
# from trl import SFTTrainer

def get_model_and_tokenizer(model_id):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
  )
  model = AutoModelForCausalLM.from_pretrained(
      model_id, quantization_config=bnb_config, device_map="auto"
  )
  model.config.use_cache=False
  model.config.pretraining_tp=1
  return model, tokenizer

def main(args):
    cfg = OmegaConf.load(args.config_file)
    df_wa = get_dataframe(cfg.data.dataset, cfg.data.source)
    _formating_prompt = partial(create_prompt, df=df_wa, 
                                task_type=cfg.task,
                                prompt_type="train"
                                )
    
    prompts = df_wa.apply(_formating_prompt, axis=1)
    
    START_TIME = time.strftime("%Y%m%d_%H%M%S")
    model_id=cfg.model.base_model
    model_name = os.path.split(cfg.model_name)[1]
    output_model=f"models/{cfg.task}/{START_TIME}/{model_name}"
    model, tokenizer = get_model_and_tokenizer(model_id)

    dataset = Dataset.from_dict({"text": prompts.to_list()})
    # dataset_dict = dataset.train_test_split(test_size=cfg.data.test_size, seed=42)
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        tokenized["labels"] = tokenized["input_ids"].copy()  # Set labels as input_ids
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=cfg.data.test_size, seed=42)

    peft_config = LoraConfig(
        r=cfg.lora.rank, 
        lora_alpha=cfg.lora.alpha, 
        lora_dropout=cfg.lora.lora_dropout, 
        bias=cfg.lora.bias, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    training_arguments = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=cfg.training.n_epochs,
        fp16=True,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
    )

    trainer.train()
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
    
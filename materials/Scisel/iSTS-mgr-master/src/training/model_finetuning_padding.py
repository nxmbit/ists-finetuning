import torch
import random
from argparse import ArgumentParser
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset, IterableDataset
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import os
import time
from src.prompt_modeling import create_prompt
from src.data.data import get_dataframe
from functools import partial
from sklearn.model_selection import train_test_split

class DatasetWrapper(IterableDataset):
    def __init__(self, texts, tokenizer, max_tokens=512):
        self.texts = texts
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.buffer = []
        self.current_token_count = 0

    def __iter__(self):
        buffer = []
        random.shuffle(self.texts)
        for text in self.texts:
            buffer += self.tokenizer(text)["input_ids"]
            buffer += [self.tokenizer.eos_token_id]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


class CustomDataset:
    def __init__(self, data, tokenizer, max_tokens):
        self.data = data
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        encoding = self.tokenizer(
            sample,
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        return encoding  # Remove batch dimension


class Trainer:
    def __init__(self, config, train_data, val_data):
        self.cfg = config
        self.load_lora = self.cfg['model']['load_lora']
        self.grad = self.cfg['training']['gradient_accumulation_steps']
        self.step = 0
        self.n_epochs = self.cfg['training']['n_epochs']
        self.lr = self.cfg['training']['learning_rate']
        self.weight_decay = self.cfg['training']['weight_decay']
        self.batch_size = self.cfg['training']['batch_size']
        lora_config = LoraConfig(
            bias =self.cfg['lora']['bias'],
            lora_alpha=self.cfg['lora']['alpha'],
            lora_dropout=self.cfg['lora']['lora_dropout'],
            r=self.cfg['lora']['rank'],
            target_modules= self.cfg['lora']['target_modules']
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model']['base_model'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

        train_dataset = CustomDataset(train_data, self.tokenizer, self.cfg['tokenizer']['max_tokens'])
        val_dataset = CustomDataset(val_data, self.tokenizer, self.cfg['tokenizer']['max_tokens'])

        # self.tokenizer = self.dataset.tokenizer
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg['training']['batch_size'],
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=32,
        )
        self.scaler = torch.cuda.amp.GradScaler()
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model']['model_to_finetune'],
        )

        if self.load_lora:
            self.model = model = get_peft_model(base_model, lora_config)
        else:
            self.model = model = base_model

        self.model = self.model.cuda()
        # self.show_params()

        # self.opt = bnb.optim.Lion(
        #     params=model.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     betas=(0.9, 0.95),
        #     optim_bits=8,
        #     # fused=True,
        # )
        self.opt = AdamW(
            params=model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.opt,
            num_warmup_steps=0,
            num_training_steps=self.n_epochs * len(self.train_loader),
        )
        self.model = torch.compile(model)

    def show_params(self):
        model = self.model
        params = sum(p.numel() for p in model.parameters())
        emb_params = list(model.gpt_neox.embed_in.parameters())
        emb_params += list(model.embed_out.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params:", params - emb_params)
        print("Params (incl. embeddings):", params)
        print("Trainable params:", trainable_params)

    def train_step(self, batch):
        batch = batch.cuda()
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", enabled=True):
            z = self.model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        self.scaler.scale(loss / self.grad).backward()
        return loss
    
    def eval_step(self, batch):
        batch = batch.cuda()
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.no_grad():
            z = self.model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)

        return loss

    def train(self):

        run_name = f'{self.cfg['wandb']['run_name']}-{str(int(time.time()))}'

        wandb.init(
            project=self.cfg['wandb']['project_name'],
            entity="albert-scisel-warsaw-university-of-technology",
            name=run_name,
            config = self.cfg
        )

        for epoch in range(1, self.n_epochs+1):
            print('\nEpoch: ', epoch)

            prog = tqdm(self.train_loader)
            self.opt.zero_grad()

            self.model.train()
            for i, batch in enumerate(prog):
                self.step += 1

                loss = self.train_step(batch)
          
                prog.set_description(f"loss: {loss.item():.3f}, step: {self.step}")
                wandb.log(
                    {
                        "loss": loss.item(),
                    },
                    step=self.step,
                )

                # if (i + 1) % self.grad == 0:
                if (self.step + 1) % self.grad == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
            

            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            total_val_batches = 0

            with torch.no_grad():  # Disable gradient calculations
                val_prog = tqdm(self.val_loader, desc="Validating")

                for i, val_batch in enumerate(val_prog):
                    loss = self.eval_step(val_batch)
                    val_loss += loss.item()
                    total_val_batches += 1
                    val_prog.set_description(f"Validation Loss: {val_loss / total_val_batches:.3f}")

                # Average validation loss over all batches
                val_loss /= total_val_batches
                perplexity = torch.exp(torch.tensor(val_loss)).item()  # Calculate perplexity

            # Log validation metrics to WandB
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_perplexity": perplexity,
                },
                step=self.step,
            )
            print(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")

        dataset_name = os.path.split(self.cfg['data']['dataset'])[1]


        # self.model.save_pretrained(f"./models/pretrained/{self.cfg['data']['task']}/pythia_{run_name}_{dataset_name}_{self.cfg['data']['train_split']}{"_lora" if self.load_lora else ""}")
        # self.model.save_pretrained(f"./models/pretrained/sst2/{run_name}_{dataset_name}_forget_{"lora" if self.load_lora else ""}")
        

def main(args):
    cfg = OmegaConf.load(args.config_file)
    df_wa = get_dataframe(cfg.data.dataset, cfg.data.source)
    _formating_prompt = partial(create_prompt, df=df_wa, 
                                instruction_type=cfg.prompt.instruction_type, 
                                label_description_type=cfg.prompt.description_level, 
                                task_type=cfg.task,
                                n_examples = cfg.prompt.n_examples)
    
    prompts = df_wa.apply(_formating_prompt, axis=1)

    train_data, val_data = train_test_split(prompts, test_size=cfg.data.test_size)
    print(train_data)
    trainer = Trainer(
        config = cfg, 
        train_data=train_data, 
        val_data=val_data
    )
    trainer.train()
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
    
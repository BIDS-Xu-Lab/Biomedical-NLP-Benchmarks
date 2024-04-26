from datasets import load_dataset
import torch,os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset,disable_caching
from utils import find_all_linear_names, print_trainable_parameters
from accelerate import Accelerator
accelerator = Accelerator()
device_index = Accelerator().process_index
device_map = {"": device_index}
import os
import argparse

disable_caching()

parser = argparse.ArgumentParser(description='Process command-line arguments.')
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help='Dataset name')
parser.add_argument('-m', '--model_name', type=str, required=True,
                    help='Model name')

args = parser.parse_args()

dataset = args.dataset
model_name = args.model_name
output_dir = f"./models/{model_name.split('/')[-1]}"+'_'+dataset.split('/')[-1]
os.environ["WANDB_PROJECT"] = model_name.split('/')[-1]+'_'+dataset.split('/')[-1]


#RE
train_dataset = load_dataset("csv", data_files=[f"./data/{dataset}/sentence_level_train.csv"], split="train")
eval_dataset = load_dataset("csv", data_files=[f"./data/{dataset}/sentence_level_dev.csv"], split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                  torch_dtype=torch.bfloat16, 
                                                  quantization_config=bnb_config,
                                                  device_map=device_map,
                                                  cache_dir="/data/yhu5/huggingface_models/",
                                                  token = token)


base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    #target_modules=find_all_linear_names(base_model),
    target_modules= [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "gate_proj",
        "up_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)
#base_model = accelerator.prepare(base_model)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['conversations'])):
        text = f"{example['conversations'][i][0]['value']} {example['conversations'][i][1]['value']}"
        output_texts.append(text)
    #print (output_texts)
    return output_texts


# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing = True,
    max_grad_norm= 0.3,
    num_train_epochs=3, 
    learning_rate=1e-5,
    bf16=True,
    save_strategy="epoch",
    save_total_limit=10,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    weight_decay=0.00001,
    warmup_ratio=0.01,
    ddp_find_unused_parameters=False,
    evaluation_strategy="epoch",
    #load_best_model_at_end=True,
    #metric_for_best_model='eval_loss'
)


trainer = SFTTrainer(
    base_model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    max_seq_length=1500,
    formatting_func=formatting_prompts_func,
    args=training_args,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

#trainer.train(resume_from_checkpoint=True) 
trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

# MODEL_NAME = "gpt2"
# MODEL_NAME = "facebook/opt-350m"
MODEL_NAME = "distilroberta-base"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

dataset_name = "Anthropic/hh-rlhf"
train_dataset = load_dataset(dataset_name, split="train[:1000]")

def preprocess_func(examples):
    new_examples={
        "input_ids_chosen":[],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": []
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen=tokenizer(chosen)
        tokenized_rejected=tokenizer(rejected)
        
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
    return new_examples

max_length = 512
train_dataset = train_dataset.map(preprocess_func, batched=True, num_proc=1)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
)

model = get_peft_model(model, peft_config)

training_args = RewardConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_dir="logs",
    logging_steps=10,
    save_steps=10,
    output_dir="output",
    overwrite_output_dir=True,
    max_length=512,
)

trainer = RewardTrainer( # TODO: I think we'll need to use another trainer if you're just going to predict the ratings for each response
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    peft_config=peft_config
)

trainer.train()


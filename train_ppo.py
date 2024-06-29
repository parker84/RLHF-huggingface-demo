from datasets import load_dataset
import torch

dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])


from trl import PPOConfig

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1
)

from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token




from transformers import pipeline

reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample

max_length = 250

dataset = dataset.map(tokenize, batched=False)
dataset = dataset.filter(
    lambda x: len(x["input_ids"]) <= 250
)


from trl import PPOTrainer

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer
)



generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_length": max_length
}


from tqdm import tqdm


epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    # for batch in tqdm(ppo_trainer.dataloader): 
    for batch in tqdm(dataset, "batch: "):
        query_tensors = torch.tensor(batch["input_ids"], device='cuda')
    
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        print(pipe_outputs)
        rewards = [-torch.tensor(output["score"]) for output in pipe_outputs]

        #### Run PPO step
        stats = ppo_trainer.step([query_tensors], [response_tensors[0]], rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_pretrained("my_ppo_model")



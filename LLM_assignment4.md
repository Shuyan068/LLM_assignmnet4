```python
# Extract the Lima dataset’s instruction
from datasets import load_dataset

lima = load_dataset("GAIR/lima")
```


```python
import json
parsed_conversations = [
    json.loads(conv) if isinstance(conv, str) else conv 
    for conv in lima["train"]["conversations"]
]
```


```python
instructions = []
for conv in parsed_conversations:
    try:
        instructions.append(conv[0]["content"])
    except (KeyError, IndexError, TypeError):
        instructions.append("")  # 跳过无效条目或标记为空
```


```python
#Sample 50 instructions
from datasets import Dataset
sampled_dataset = Dataset.from_dict({"instruction": instructions}).shuffle(seed=42).select(range(50))
```


```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```


```python
# use mistralai/Mistral-7B-Instruct-v0.2 to generate 5 responses for each instruction
import torch
from tqdm import tqdm

all_responses = []
for instruction in tqdm(sampled_dataset["instruction"]):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        return_tensors="pt"
    ).to(model.device)
    
    # 生成5个不同回答
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=5,
        do_sample=True
    )
    responses = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    all_responses.append(responses)
```


```python
#use PairRM to create a preference dataset
from pairrm import PairRMScorer

scorer = PairRMScorer()
preference_data = []

for idx, instruction in tqdm(enumerate(sampled_dataset["instruction"])):
    candidates = all_responses[idx]
    pairs = [(i, j) for i in range(5) for j in range(5) if i < j]
    scores = []
    
    for i, j in pairs:
        score = scorer.score([instruction], [candidates[i]], [candidates[j]])
        scores.append((i, j, score))
    
    best_idx = max(scores, key=lambda x: x[2])[0]
    worst_idx = min(scores, key=lambda x: x[2])[1]
    preference_data.append({
        "instruction": instruction,
        "chosen": candidates[best_idx],
        "rejected": candidates[worst_idx],
    })

from datasets import Dataset

dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in preference_data],
    "chosen": [d["chosen"] for d in preference_data],
    "rejected": [d["rejected"] for d in preference_data],
})
# Push this dataset to huggingface
dataset.push_to_hub("ShuyanCHEN/DSAA6000_assignment4")
```


```python
#Use DPO to fine tune mistralai/Mistral-7B-Instruct-v0.2
from transformers import TrainingArguments
from trl import DPOTrainer
import pandas as pd
import torch

#sample 10 instructions that were not seen in training and generate samples
dpo_dataset = dataset.train_test_split(
    test_size=0.2,  
    shuffle=True,
    seed=42
)
```


```python
training_args = TrainingArguments(
    per_device_train_batch_size=2,    
    gradient_accumulation_steps=2,    
    learning_rate=5e-6,                
    num_train_epochs=2,               
    logging_steps=10,
    evaluation_strategy="no",
    output_dir="./dpo_model",
    fp16=True,                       
    optim="adamw_torch",
    report_to="none"                 
)
```


```python
#Compare the completions from the original model (mistralai/Mistral-7B-Instruct-v0.2 and your DPO fine tuned model
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,                    
    args=training_args,
    train_dataset=dpo_dataset["train"],
    tokenizer=tokenizer,
    beta=0.1,                        
    max_length=512,
)
dpo_trainer.train()
```


```python

test_instructions = dpo_dataset["test"]["instruction"][:10] 

original_outputs = []
dpo_outputs = []


def generate_response(instruction, model):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


for instr in test_instructions:

    original_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", 
        device_map="auto"
    )
    original_outputs.append(generate_response(instr, original_model))
    

    dpo_outputs.append(generate_response(instr, dpo_trainer.model))

```




```python
# Display the instruction, original model completion, and DPO fine-tuned model completion as a pandas dataframe
df = pd.DataFrame({
    "Instruction": test_instructions,
    "Original Model": original_outputs,
    "DPO Model": dpo_outputs,
})

pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 1000)

#print out the dataframe to stdout
print(d![img.png](img.png)f.to_markdown(index=False))


```


```python
#Push the PEFT adapter to huggingface
dpo_trainer.model.save_pretrained("mistral-7b-dpo-adapter")

from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="mistral-7b-dpo-adapter",
    repo_id="ShuyanCHEN/DSAA6000_assignment4",
    repo_type="model"
)

print("\nhttps://huggingface.co/ShuyanCHEN/DSAA6000_assignment4")

```

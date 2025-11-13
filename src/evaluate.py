import json
import random
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

# Load model + tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("aws/package/model")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Read all prompts
with open("data_gen/training/eval_prompts.jsonl") as f:
    lines = f.readlines()

# Sample N random prompts (e.g. 50 out of 1700)
sample_size = 50
sampled_lines = random.sample(lines, sample_size)

# Evaluate only the sampled prompts
for line in sampled_lines:
    prompt = json.loads(line)["input"]
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"### Prompt\n{prompt}\n")
    print(f"**Suggestion:**\n{suggestion}\n")
    print("---\n")

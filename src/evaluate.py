import json
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("aws/package/model")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

with open("data/eval_prompts.jsonl") as f:
    for line in f:
        prompt = json.loads(line)["input"]
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=128)
        print("Prompt:", prompt)
        print("Suggestion:", tokenizer.decode(outputs[0], skip_special_tokens=True))

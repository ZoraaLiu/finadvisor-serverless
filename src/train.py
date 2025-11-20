from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, T5Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch

# Load model + tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Load dataset
dataset = load_dataset("json", data_files="data_gen/training/instructions.jsonl", split="train")

def preprocess(example):
    model_inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            truncation=True,
            padding="max_length",
            max_length=128
        )["input_ids"]

    # Replace pad tokens with -100
    labels = [id if id != tokenizer.pad_token_id else -100 for id in labels]
    model_inputs["labels"] = labels
    return model_inputs


tokenized = dataset.map(preprocess)

# debug: check one sample
model = model.to("cuda")

batch = tokenized[0]
input_ids = torch.tensor([batch["input_ids"]]).to("cuda")
labels = torch.tensor([batch["labels"]]).to("cuda")

outputs = model(input_ids=input_ids, labels=labels)
print("Manual loss:", outputs.loss.item())


# LoRA config
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training
training_args = TrainingArguments(
    output_dir="aws/package/model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,              # log every step
    logging_first_step=True,      # log step 0
    save_steps=500,
    save_total_limit=2,
    fp16=False,                    # keep mixed precision for speed
    report_to="none",
    eval_strategy="no",  # add eval logging
    # eval_steps=50
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()

# Explicitly save model + tokenizer so evaluation works
trainer.save_model("aws/package/model")
tokenizer.save_pretrained("aws/package/model")

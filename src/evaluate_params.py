import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    # Load base model + tokenizer
    model_name = "google/flan-t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Print parameter counts before LoRA
    total, trainable = count_parameters(model)
    print("=== Base Model ===")
    print(f"Total parameters: {total/1e6:.2f}M")
    print(f"Trainable parameters: {trainable/1e6:.2f}M")
    print(f"Trainable %: {100 * trainable / total:.2f}%\n")

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    # Print parameter counts after LoRA
    total, trainable = count_parameters(model)
    print("=== LoRA Model ===")
    print(f"Total parameters: {total/1e6:.2f}M")
    print(f"Trainable parameters: {trainable/1e6:.2f}M")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

    # Optional: print trainable parameter names
    print("\nTrainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}: {param.numel()/1e3:.1f}K params")

if __name__ == "__main__":
    main()

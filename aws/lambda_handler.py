from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/opt/model")
model = AutoModelForSeq2SeqLM.from_pretrained("/opt/model")

def lambda_handler(event, context):
    input_text = event["input"]
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"suggestions": suggestion}

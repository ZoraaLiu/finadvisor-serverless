import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load model and tokenizer once (cold start)
MODEL_NAME = "google/flan-t5-small"
ADAPTER_PATH = "./model"  # local folder inside your Lambda ZIP

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        user_input = body.get("query", "")

        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=256)

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"response": response_text})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

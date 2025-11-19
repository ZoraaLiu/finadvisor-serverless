import os
# Must be set before importing transformers
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_NAME = "google/flan-t5-small"
ADAPTER_PATH = "./model"

# Now load tokenizer and model; they will cache into /tmp
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

def lambda_handler(event, context):
    try:
        print("EVENT:", event)

        body_str = event.get("rawBody") or event.get("body")
        if isinstance(body_str, dict):
            body = body_str
        elif isinstance(body_str, str):
            body = json.loads(body_str)
        else:
            body = {}

        user_input = body.get("query", "")
        if not user_input:
            raise ValueError("Missing 'query' in request body")

        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=256)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"response": response_text})
        }
    except Exception as e:
        print("ERROR:", str(e))
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }

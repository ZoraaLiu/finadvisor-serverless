# finadvisor-serverless

A lightweight, serverless financial advisory system powered by a fine-tuned small language model (LLM) deployed on AWS Lambda. The system analyzes simulated personal financial transaction data to provide personalized budgeting and saving suggestions. This project demonstrates how machine learning models can run efficiently on serverless infrastructure with minimal latency and resource overhead.

## Project Overview

**Core Components:**
- **Data Generation Pipeline**: Synthetic transaction data generation and instruction/evaluation prompt building
- **Model**: Fine-tuned Google FLAN-T5-Small with LoRA adapters for parameter-efficient training
- **Lambda Backend**: Serverless API endpoint that loads the model and generates financial advice
- **Web UI**: Simple HTML interface for querying the financial advisor

**Key Features:**
- Generates realistic multi-user, multi-month financial transaction data
- Creates instruction-following pairs for model fine-tuning
- Fine-tunes FLAN-T5 using LoRA to achieve >60% parameter efficiency
- Deploys as AWS Lambda with Docker containerization
- Provides fast inference (sub-second latency) on serverless infrastructure

---

## Dependencies

### Core Dependencies (pyproject.toml)
```
python>=3.10
pandas^2.2.2        # Data manipulation
numpy^1.26.4        # Numerical computing
faker^25.0.0        # Synthetic data generation
pyyaml^6.0.3        # YAML config file parsing
```

### Model Training & Inference (Lambda Dockerfile)
```
numpy==1.26.4
torch==2.0.1+cpu
transformers==4.39.3    # HuggingFace model library
peft==0.8.2             # Parameter-efficient fine-tuning (LoRA)
sentencepiece==0.1.99   # Tokenizer for FLAN-T5
```

### Additional Development Dependencies
- `datasets` - HuggingFace datasets library (for loading training data)
- `psutil` - System monitoring (for test performance metrics)

---

## Project Structure

```
finadvisor-serverless/
├── data_gen/                          # Data generation pipeline
│   ├── generate_transactions.py        # Synthetic transaction generator
│   ├── make_instructions.py            # Instruction/evaluation prompt builder
│   ├── summaries.py                    # Financial summary utilities
│   ├── baseline.py                     # Baseline budgeting suggestions
│   ├── pipeline.py                     # Main orchestration script
│   ├── configs/
│   │   ├── generate.yaml               # Transaction generation config
│   │   └── instructions.yaml           # Instruction generation config
│   └── data/
│       └── transactions.csv            # Generated synthetic transactions
│
├── src/                               # Source scripts
│   ├── train.py                        # Model fine-tuning script
│   ├── evaluate.py                     # Model evaluation on test set
│   ├── evaluate_params.py              # LoRA parameter analysis
│   └── baseline.py                     # Baseline suggestion logic
│
├── lambda_package/                    # AWS Lambda deployment
│   ├── lambda_function.py              # Lambda handler entry point
│   ├── test_lambda.py                  # Local testing suite
│   └── model/                          # Fine-tuned LoRA adapter
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── tokenizer_config.json
│       ├── spiece.model
│       └── checkpoint-*/               # Training checkpoints
│
├── layer_torch/                       # (Optional) Separated torch layer
├── layer_transformers/                # (Optional) Separated transformers layer
│
├── pyproject.toml                     # Poetry dependencies
├── Dockerfile                         # Lambda container image
├── ui.html                            # Web UI for testing
├── load_test.html                     # Load testing interface
├── compare_trained_and_untrained_model_results.ipynb
├── finetune_flan_t5.ipynb             # Training notebook
└── README.md                          # This file
```

---

## File & Folder Descriptions

### `data_gen/` - Data Generation Pipeline

**Purpose**: Generate synthetic financial data and create training/evaluation datasets.

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestration script; chains generation → instruction building |
| `generate_transactions.py` | Creates N users with synthetic transactions over M months from YAML config |
| `make_instructions.py` | Builds instruction-following pairs (input prompts → budgeting suggestions) |
| `summaries.py` | Calculates financial summaries (monthly spend, category breakdowns, trends) |
| `baseline.py` | Provides rule-based budgeting suggestions (fallback/baseline) |
| `configs/generate.yaml` | Config for transaction generation (user count, merchants, amounts, etc.) |
| `configs/instructions.yaml` | Config for instruction generation (time windows, split ratios) |
| `data/transactions.csv` | Output CSV with all generated transactions |

### `src/` - Training & Evaluation

| File | Purpose |
|------|---------|
| `train.py` | Fine-tunes FLAN-T5-Small with LoRA on instructions.jsonl; saves to `aws/package/model` |
| `evaluate.py` | Loads trained model and runs inference on sampled eval prompts |
| `evaluate_params.py` | Analyzes LoRA adapter parameters (total, trainable, efficiency %) |
| `baseline.py` | Utility for baseline suggestions |

### `lambda_package/` - AWS Lambda Deployment

**Purpose**: Production inference endpoint.

| File | Purpose |
|------|---------|
| `lambda_function.py` | Lambda handler; loads model and tokenizer on cold start, generates responses |
| `test_lambda.py` | Local testing: latency, reliability, throughput, memory profiling |
| `model/` | Fine-tuned LoRA adapter weights and tokenizer |

### `ui.html` - Web Interface

Simple HTML + JavaScript interface to query the financial advisor via the Lambda API endpoint.

### Other Files

- `Dockerfile` - Container image for AWS Lambda (Python 3.9 + torch + transformers + peft)
- `pyproject.toml` - Poetry project config and dependencies
- `load_test.html` - (Optional) Load testing interface
- `*.ipynb` - Jupyter notebooks for experimentation and comparison

---

## Installation & Setup

### 1. Install Poetry (if not already installed)
```powershell
# On Windows
Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing | python -
```

### 2. Clone the repository and install dependencies
```powershell
cd finadvisor-serverless
poetry install
```

This installs all dependencies from `pyproject.toml` into a virtual environment.

### 3. Verify Installation
```powershell
poetry run python --version
```

---

## Data Generation Steps

The data generation pipeline has two stages: **transaction generation** → **instruction building**.

### Stage 1: Generate Synthetic Transactions

**Config file**: `data_gen/configs/generate.yaml`

This YAML file defines:
- **Users**: Count (100), currencies (HKD/USD/SGD), income range (2000-38000)
- **Merchants**: Categorized by spending category
- **Transactions**: Span (24 months), rent range, subscriptions, variable spending patterns
- **Categories**: 9 categories (Rent, Groceries, Dining, Transport, Utilities, Shopping, Entertainment, Health, Other)

**Output**: `data_gen/data/transactions.csv`

### Stage 2: Build Instructions & Evaluation Prompts

**Config file**: `data_gen/configs/instructions.yaml`

This YAML defines:
- **Output splits**: 75% training, 25% evaluation
- **Time windows**: 12 "as-of" dates (weekly intervals) to generate prompts
- **Analysis periods**: "week", "month", "year" + custom day ranges (45, 90, 180)

**Outputs**:
- `data_gen/training/instructions.jsonl` - Training data (input prompts → output suggestions)
- `data_gen/training/eval_prompts.jsonl` - Evaluation prompts only

### Run the Full Pipeline

```powershell
cd data_gen
poetry run python pipeline.py --all
```

**Breakdown:**
```powershell
# Generate transactions only
poetry run python pipeline.py --generate

# Build instructions only (requires transactions.csv)
poetry run python pipeline.py --prompts

# Full pipeline (both steps)
poetry run python pipeline.py --all
```

**Options:**
```powershell
# Use existing CSV instead of generating new one
poetry run python pipeline.py --prompts --csv path/to/existing.csv

# Custom config paths
poetry run python pipeline.py --all \
  --gen-config custom_gen.yaml \
  --instr-config custom_instr.yaml \
  --out-csv custom_output.csv
```

---

## Model Training & Evaluation

### Prerequisites
- Install PyTorch and transformers (may need GPU/CUDA for faster training):
  ```powershell
  poetry add torch transformers peft datasets
  ```

### Model Architecture

**Base Model**: Google FLAN-T5-Small
- ~76.96M parameters
- Seq2Seq (encoder-decoder) architecture
- Pre-trained on diverse instruction-following tasks

**Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- Rank (r): 8
- Alpha: 16
- Dropout: 0.1
- **Result**: ~97% parameter freeze, only ~0.34M trainable parameters (~0.45% of total)

### Training

```powershell
cd src
python train.py
```

**Script behavior**:
1. Loads FLAN-T5-Small base model + tokenizer
2. Loads training data from `data_gen/training/instructions.jsonl`
3. Preprocesses: tokenizes input/output with padding to 256/128 tokens
4. Applies LoRA config to model
5. Trains with:
   - Batch size: 8
   - Epochs: 3
   - Logging every 50 steps
   - Checkpoints saved every 500 steps (keeps last 2)
6. Saves final model to `aws/package/model`

**Output**:
```
aws/package/model/
├── adapter_model.safetensors  # LoRA weights
├── adapter_config.json
├── tokenizer_config.json
├── special_tokens_map.json
├── spiece.model
└── checkpoint-*/              # Training checkpoints
```

### Evaluation

```powershell
cd src
python evaluate.py
```

**Script behavior**:
1. Loads trained model from `aws/package/model`
2. Loads evaluation prompts from `data_gen/training/eval_prompts.jsonl`
3. Randomly samples 50 prompts
4. Generates responses using `model.generate(max_length=128)`
5. Displays prompt → suggestion pairs

**Example output**:
```
### Prompt
Profile: single user, stable spender. Totals: Income SGD 200,000; Spend SGD 120,000; Net SGD 80,000. ...

**Suggestion:**
Consolidate subscriptions and review dining frequency. Consider setting a monthly dining budget of SGD 2,000...

---
```

### Parameter Analysis

```powershell
cd src
python evaluate_params.py
```

**Output**:
```
=== Base Model ===
Total parameters: 80.19M
Trainable parameters: 80.19M
Trainable %: 100.00%

=== LoRA Model ===
Total parameters: 80.46M
Trainable parameters: 0.30M
Trainable %: 0.37%
```

---

## Lambda Function & Deployment

### Local Testing

```powershell
cd lambda_package
python test_lambda.py
```

**Tests**:
- **Latency**: Single request time
- **Reliability**: Handles malformed JSON gracefully
- **Throughput**: 10 concurrent requests
- **Memory**: Approximate memory usage

**Example output**:
```
Response: {'statusCode': 200, 'body': '{"response": "..."}'}
Latency: 0.45 seconds
Throughput: 15.2 requests/sec
Memory usage: 2048 MB
```

### Lambda Function Handler (`lambda_function.py`)

**Key features**:
- Sets Hugging Face cache to `/tmp/` for Lambda's ephemeral storage
- Loads base model + LoRA adapter on cold start
- Accepts POST requests with JSON body: `{"query": "..."}`
- Returns JSON response: `{"response": "..."}`
- Handles errors gracefully

**Environment variables** (set in Lambda console):
```
TRANSFORMERS_CACHE=/tmp/huggingface
HF_HOME=/tmp/huggingface
HF_DATASETS_CACHE=/tmp/huggingface/datasets
```

### Docker Deployment

```powershell
# Build image
docker build -t finadvisor:latest .

# Run locally
docker run -p 9000:8080 finadvisor:latest

# Test
$body = @{ query = "I have HKD 10,000 monthly income..." } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:9000/2015-03-31/functions/function/invocations `
  -Method POST -Body $body -ContentType application/json
```

### Deploy to AWS Lambda

```powershell
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker build -t finadvisor:latest .
docker tag finadvisor:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/finadvisor:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/finadvisor:latest

# Create/update Lambda function from ECR image
aws lambda create-function \
  --function-name finadvisor-advisor \
  --role arn:aws:iam::123456789:role/lambda-role \
  --code ImageUri=123456789.dkr.ecr.us-east-1.amazonaws.com/finadvisor:latest \
  --timeout 60 \
  --memory-size 3008
```

---

## Web Application Usage

### Local Testing

```powershell
# Start HTTP server
python -m http.server 8080
```

Open browser: `http://localhost:8080/ui.html`

**Interface**:
- Text area for entering financial query
- Submit button to send to Lambda
- Displays response from LLM

### Example Queries

```
Profile: comfortable renter. Totals: Income SGD 338,100; Spend SGD 219,176; Net SGD 118,924. Top spend: Rent:117,108, Shopping:24,642, Utilities:19,279. Recurring: SGD 117,984 across 1 items. Task: Suggest 2-3 actionable steps to reduce spend with ~SGD monthly savings estimates.
```

### Production URL

The current Lambda endpoint is deployed at:
```
https://1lv2h7f7mi.execute-api.us-east-1.amazonaws.com/prod/predict
```

(As configured in `ui.html`)

---

## Configuration Files

### `data_gen/configs/generate.yaml`

Defines synthetic transaction generation parameters:

```yaml
random_seed: 42                        # Reproducibility
np_seed: 42

categories: [Rent, Groceries, ...]    # Spending categories

merchants:                             # Merchants per category
  Rent: ["Acme Property", ...]
  Groceries: ["ParknShop", ...]
  ...

users:
  count: 100                           # Number of simulated users
  currencies: [HKD, USD, SGD]
  income_range: [2000, 38000]

transactions:
  months: 24                           # Transaction history span
  rent_range: [4000, 16000]
  subscriptions:
    count_range: [1, 2]
    amount_range: [60, 160]
    merchants: [...]
  variable_spend:
    per_month_range: [20, 40]          # Num transactions per month
    category_amounts: {...}            # Min/max per category
```

### `data_gen/configs/instructions.yaml`

Defines instruction dataset generation:

```yaml
random_seed: 42

outputs:
  train: training/instructions.jsonl   # Training data
  eval: training/eval_prompts.jsonl    # Evaluation prompts

split:
  train_frac: 0.75                     # 75% train, 25% eval

users_limit: null                      # Limit to N users (null = all)

as_of:                                 # Time windows for prompts
  count: 12
  step_days: 7                         # Weekly intervals

periods: [week, month, year]           # Analysis periods

custom_days: [45, 90, 180]             # Additional custom windows
```

---

## Workflow Summary

### Complete End-to-End Flow

```
1. Generate synthetic data
   └─ poetry run python data_gen/pipeline.py --all
   
2. Fine-tune model on generated instructions
   └─ python src/train.py
   
3. Evaluate model performance
   └─ python src/evaluate.py
   
4. Test Lambda locally
   └─ python lambda_package/test_lambda.py
   
5. Run web UI locally
   └─ python -m http.server 8080
   
6. Deploy to AWS Lambda
   └─ docker build . && aws lambda update-function-code ...
   
7. Query via web UI
   └─ Open ui.html in browser or use API endpoint
```
---

## References

- [Google FLAN-T5](https://huggingface.co/google/flan-t5-small)
- [PEFT / LoRA](https://github.com/huggingface/peft)
- [AWS Lambda with Docker](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

## Author

NG Sum Yu (Sammi), LIU Shixiao (Zora), POON Tsz Hang (Harry)

The Hong Kong University of Science and Technology

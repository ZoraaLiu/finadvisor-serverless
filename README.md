# finadvisor-serverless
A lightweight financial advisory system powered by a small language model (LLM) deployed on AWS Lambda. The system analyzes simulated user spending data to provide personalized budgeting and saving suggestions, demonstrating how machine learning models can run efficiently on serverless infrastructure.

# Setup

## Install Dependencies
Use `Poetry`￼for environment management:
```
poetry install
```

## Data Gen
Use `poetry run python pipeline.py --all` to run the full pipeline and generate dataset
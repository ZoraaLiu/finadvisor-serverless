# Base image: AWS Lambda Python runtime
FROM public.ecr.aws/lambda/python:3.9

# Install dependencies
RUN pip install torch transformers peft --no-cache-dir

# Copy your Lambda package into the container
COPY lambda_package/ ${LAMBDA_TASK_ROOT}/

# Set handler (points to lambda_function.py inside lambda_package)
CMD ["lambda_function.lambda_handler"]

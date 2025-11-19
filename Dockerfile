FROM public.ecr.aws/lambda/python:3.9

# Install dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    torch==2.0.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    transformers==4.39.3 \
    peft==0.8.2 \
    sentencepiece==0.1.99

# Copy everything from lambda_package into Lambda task root
COPY lambda_package/ ${LAMBDA_TASK_ROOT}/

# Set the handler (lambda_function.py inside lambda_package)
CMD ["lambda_function.lambda_handler"]
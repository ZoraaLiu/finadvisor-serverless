import time
import concurrent.futures
import json
import psutil
from lambda_function import lambda_handler

# Simulated Lambda event
event = {
    "body": '{"query": "Profile: comfortable renter. Totals: Income SGD 338,100; Spend SGD 219,176; Net SGD 118,924. Top spend: Rent:117108, Shopping:24642, Utilities:19279. Recurring: SGD 117,984 across 1 items. Delivery orders (approx): 19. Task: Suggest 2-3 actionable steps to reduce spend with ~SGD monthly savings estimates."}'
}
context = None  # not needed for basic testing

# Latency
start = time.time()
response = lambda_handler(event, context)
end = time.time()
print("Response:", response)
print("Latency:", end - start, "seconds")

# Reliability test (bad input)
bad_event = {"body": "not-json"}
print("Bad input response:", lambda_handler(bad_event, context))

# Throughput test (simulate 10 concurrent requests)
def run_request(i):
    return lambda_handler(event, context)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    start = time.time()
    results = list(executor.map(run_request, range(10)))
    end = time.time()
    print("Throughput:", 10 / (end - start), "requests/sec")

# Memory usage (approximate, local only)
process = psutil.Process()
print("Memory usage:", process.memory_info().rss / (1024*1024), "MB")

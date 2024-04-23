from openai import OpenAI
import os
import requests
from tqdm import trange
from time import time

# modify this value to match your host, remember to add /v1 at the end

# VLLM_HOST = "http://0.0.0.0:8000"
VLLM_HOST = "http://0.0.0.0:5000"

url = f"{VLLM_HOST}/v1/completions"
print(url)
headers = {"Content-Type": "application/json"}

def prompt(query, model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"):
    data = {
        "model": model,
        "prompt": query,
        "max_tokens": 100,
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# model = "facebook/opt-125m"
# model = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
model = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

# start = time()
# for i in trange(10):
#     print(prompt("JupySQL is", model=model))
# print(time()-start)
# print()

openai_api_base = "http://0.0.0.0:5000/v1"
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
        base_url=openai_api_base,
        api_key=openai_api_key
    )


start = time()
for i in trange(10):
    completion = client.completions.create(model=model,
                                      prompt="JupySQL is",
                                      max_tokens=50)
    print(completion.choices[0].text)
print(time()-start)
print()

start = time()
for i in trange(10):
    completion = client.chat.completions.create(model=model,
                                      messages=[{"role": "user", "content": "JupySQL is"}],
                                      max_tokens=50)
    print(completion.choices[0].message.content)
print(time()-start)
print()

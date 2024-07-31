import os
from openai import AzureOpenAI, OpenAI

def init_openai_client(port=None, personal=False):
    openai_api_key = os.getenv("OPENAI_API_KEY") if not personal else os.getenv("OPENAI_API_KEY_ABE")
    if port != None:
        openai_api_base = f"http://0.0.0.0:{port}/v1"
        return OpenAI(
            base_url=openai_api_base,
            api_key=openai_api_key
        )
    return OpenAI(
        api_key=openai_api_key
    )


def init_azure_openai_client():
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    return AzureOpenAI(
        api_key=azure_openai_api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_openai_endpoint
    )

def request_azure_generate(client, prompt, max_retries = 10, max_tokens=200, system_prompt="Assistant is a large language model trained by OpenAI."):
    num_try = 0
    while num_try < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt4v", # model = "deployment_name".
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{prompt}"}
                ],
                seed = 42,
                max_tokens=max_tokens
            )
            output = response.choices[0].message.content
            if len(output.split()) < max_tokens // 10:
                raise Exception("Output is too short - maybe an error")
            # print(f"Usage: {response.usage}")
            return output
        except Exception as e:
            print(f"Error: {e}")
            num_try += 1
    


def request_GPT(client, prompt, max_retries = 10, max_tokens=5, model="gpt-3.5-turbo-instruct", system_prompt=None):
    num_try = 0
    # print(prompt)
    while num_try < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{prompt}"}
                ],
                max_tokens=max_tokens,
                seed = 42
            )
            if type(prompt) == list:
                return [r.text for r in response.choices]
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            num_try += 1

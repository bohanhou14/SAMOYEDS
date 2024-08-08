from engines.engine import Engine
import multiprocessing as mp
import os
import time
from openai import OpenAI
from utils.utils import clean_response, parse_attitude, compile_enumerate, parse_actions, counter_to_ordered_list, HESITANCY
from collections import Counter
from sandbox.tweet import Tweet
from sandbox.prompts import *
from tqdm import trange
from utils.utils import ATTITUDES

class DataParallelEngine(Engine):
    def __init__(self, num_processes=10, ports=None, batch_size=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = num_processes
        self.ports = ports if ports else [8000, 7000, 6000, 5000]
        self.batch_size = batch_size
    
    def init_openai_client(self, port):
        openai_api_base = f"http://0.0.0.0:{port}/v1"
        openai_api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI(
            base_url=openai_api_base,
            api_key=openai_api_key
        )
    
    def request_greedy_generate(self, prompt, port, max_tokens=80):
        try:
            client = self.init_openai_client(port)
            completion = client.chat.completions.create(
                model=self.model_type,
                messages=prompt,
                seed = self.seed,
                temperature=0.0,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
        
    def request_generate(self, prompt, port, max_tokens=80):
        try:
            client = self.init_openai_client(port)
            completion = client.chat.completions.create(
                model=self.model_type,
                messages=prompt,
                max_tokens=max_tokens,
                top_p=1.0,
                seed=self.seed
            )
            return completion.choices[0].message.content
        except Exception as e:
            # print(prompt)
            return f"Error: {e}"

    def request_generate_attitude(self, prompt, port, max_tokens=80):
        attitude = None
        num_iter = 0
        while attitude not in ATTITUDES and num_iter < self.max_iter:
            response = self.request_generate(prompt, port, max_tokens=max_tokens)
            print("response", response)
            attitude = parse_attitude(response)[0]
            if attitude in ATTITUDES:
                return attitude
            else:
                num_iter += 1
        return "probably no"

    def request_generate_actions(self, prompt, port, max_tokens=80):
        action = ""
        while len(action) < 2:
            response = self.request_generate(prompt, port, max_tokens=max_tokens, temperature=1, top_p=1)
            action = clean_response(response)
        return action

    def chunkify(self, lst, n):
        """Split list into chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate(self, max_tokens):
        print(f"Stage: {self.stage}, pool map started")
        start = time.time()
        batches = list(self.chunkify(self.context, self.batch_size))
        results = []

        with mp.get_context('spawn').Pool(processes=self.num_processes) as pool:
            for i, batch in enumerate(batches):
                ports = [self.ports[j % len(self.ports)] for j in range(i * self.batch_size, (i + 1) * self.batch_size)]
                res = pool.starmap(self.request_generate, [(msg, port, max_tokens) for msg, port in zip(batch, ports)])
                results.extend(res)

        results = [clean_response(r) for r in results]
        end = time.time()
        print(f"Stage: {self.stage}, pool map finished")
        print(f"Time taken for parallel execution: {end - start}")
        return results

    def generate_attitude(self, max_tokens):
        print(f"Stage: {self.stage}, pool map started")
        start = time.time()
        batches = list(self.chunkify(self.context, self.batch_size))
        results = []

        with mp.get_context('spawn').Pool(processes=self.num_processes) as pool:
            for i, batch in enumerate(batches):
                ports = [self.ports[j % len(self.ports)] for j in range(i * self.batch_size, (i + 1) * self.batch_size)]
                res = pool.starmap(self.request_generate_attitude, [(msg, port, max_tokens) for msg, port in zip(batch, ports)])
                results.extend(res)

        end = time.time()
        print(f"Stage: {self.stage}, pool map finished")
        print(f"Time taken for parallel execution: {end - start}")
        return results

    def generate_actions(self, max_tokens):
        print(f"Stage: {self.stage}, pool map started")
        start = time.time()
        batches = list(self.chunkify(self.context, self.batch_size))
        results = []

        with mp.get_context('spawn').Pool(processes=self.num_processes) as pool:
            for i, batch in enumerate(batches):
                ports = [self.ports[j % len(self.ports)] for j in range(i * self.batch_size, (i + 1) * self.batch_size)]
                res = pool.starmap(self.request_generate_actions, [(msg, port, max_tokens) for msg, port in zip(batch, ports)])
                results.extend(res)

        end = time.time()
        print(f"Stage: {self.stage}, pool map finished")
        print(f"Time taken for parallel execution: {end - start}")
        return results
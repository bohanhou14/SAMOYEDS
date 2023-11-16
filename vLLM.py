from vllm import LLM
llm = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")
print(output)

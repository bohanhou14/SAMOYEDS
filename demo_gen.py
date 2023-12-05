
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import parse_profile
from vllm import LLM, SamplingParams
import pandas as pd
from engine import Engine
import pickle

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
state = 'Maryland'
attitude = 'hesitant'

messages = [
    {"role": "user",
     "content":
        f'''Generate a demographic profile of someone from 
            {state} that feels {attitude} about COVID vaccination.
            
            Example: 
                - Name: Garcia Marquez
                - Gender: male
                - Race: Hispanic
                - Education: High School
                - Age: 45 years old
                - Occupation: farm owner
                - Religion: atheist
                - Political belief: neutral
            
            Generate profile:
        '''
     },
#    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, tokenize = False)
model_inputs = encodeds

p = 0.7
temp = 2.0
model = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=1)
sampling_params = SamplingParams(
    top_p = p,
    temperature = temp,
    max_tokens = 512
)
profiles = []
agent_num = 500
for i in range(agent_num):
    res = model.generate(model_inputs)
    profile = parse_profile(res[0].outputs[0].text)
    if 'Gender' not in profile.keys():
        print(f"Extraction failed: {res}")
        continue
    profiles.append(profile)

with open(f'profiles/profiles-agent_num={agent_num}-top_p={p}-temp={temp}.pkl', 'wb') as f:
    pickle.dump(profiles, f)
df = pd.DataFrame(profiles)
df.to_csv(f'profiles/profiles-top_p={p}-temp={temp}.tsv', sep='\t')



# gender = [p['Gender'] for p in profiles]
# name = [p['Name'] for p in profiles]
# religion = [p['Religion'] for p in profiles]
# pb = [p['Political belief'] for p in profiles]
# age = [p['Age'] for p in profiles]
# oc = [p['Occupation'] for p in profiles]
# ed = [p['Education'] for p in profiles]



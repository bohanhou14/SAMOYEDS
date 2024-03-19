
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import parse_profile
from vllm import LLM, SamplingParams
from tqdm import trange
from engine import Engine
import pandas as pd
import pickle
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_agents", type=int)
    parser.add_argument("p", type=float)
    parser.add_argument("temp", type=float)
    parser.add_argument("--state",default="United States")
    parser.add_argument("--attitude", default='"probably no, definitely no, probably yes, definitely yes"')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    state = args.state
    attitude = args.attitude
    messages = [
        {"role": "user",
         "content":
             f'''Generate a demographic profile of someone from 
                {state} that will say {attitude} to COVID vaccination.

                Example: 
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
    model_inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    p = args.p
    temp = args.temp
    model = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=1)
    sampling_params = SamplingParams(
        top_p=p,
        temperature=temp,
        max_tokens=512
    )
    profiles = []
    agent_num = args.num_agents
    for i in trange(agent_num):
        res = model.generate(model_inputs, sampling_params)
        profile = parse_profile(res[0].outputs[0].text)
        if 'Gender' not in profile.keys():
            print(f"Extraction failed: {res[0].outputs[0].text}")
            continue
        profiles.append(profile)

    save_name = f"profiles/profiles-state-{state}=attitude={attitude}-agent_num={agent_num}-top_p={p}-temp={temp}"
    with open(f'{save_name}.pkl', 'wb') as f:
        pickle.dump(profiles, f)
    df = pd.DataFrame(profiles)
    df.to_csv(f'{save_name}.tsv', sep='\t')


# gender = [p['Gender'] for p in profiles]
# name = [p['Name'] for p in profiles]
# religion = [p['Religion'] for p in profiles]
# pb = [p['Political belief'] for p in profiles]
# age = [p['Age'] for p in profiles]
# oc = [p['Occupation'] for p in profiles]
# ed = [p['Education'] for p in profiles]



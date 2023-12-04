import pickle
from agent import Agent
from vllm import LLM, SamplingParams
import os
from collections import Counter
from engine import Engine
from utils import clean_response, parse_attitude
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("profiles/profiles-agent_num=10-top_p=0.7-temp=2.0.pkl", "rb") as f:
    # a list of dictionaries
    profiles = list(pickle.load(f))

profile_0 = profiles[0]
profile_1 = profiles[1]
profile_2 = profiles[2]

agent_0 = Agent(profile=profile_0)
profile_str_0 = agent_0.get_profile_str()
agent_1 = Agent(profile=profile_1)
profile_str_1 = agent_1.get_profile_str()
agent_2 = Agent(profile=profile_2)
profile_str_2 = agent_2.get_profile_str()


messages_0 = [
    {"role": "user",
     "content":
        f'''
            Example A:
            Pretend you are this person: 
                - Name:  Karen Williams
                - Gender:  female
                - Age:  50 years old
                - Education:  College graduate
                - Occupation:  small business owner
                - Political belief:  moderate democrat
                - Religion:  Baptist
            What's your attitude towards getting COVID vaccination? 
            Attitude: probably yes.

            Example B:
            Pretend you are this person: 
                - Name:  Ava Green
                - Gender:  female
                - Age:  27 years old
                - Education:  college degree in science
                - Occupation:  stay-at-home mom
                - Political belief:  Republican
                - Religion:  Baptist
            What's your attitude towards getting COVID vaccination? 
            Attitude: probably no.

            Pretend you are this person: {profile_str_0}\n
            Choose from definitely yes, probably yes, probably no, definitely no.
            What's your attitude towards getting COVID vaccination? 
            Attitude: 
        '''
     },
]

messages_1 = [
    {"role": "user",
     "content":
        f'''
            Example A:
            Pretend you are this person: 
                - Name:  Karen Williams
                - Gender:  female
                - Age:  50 years old
                - Education:  College graduate
                - Occupation:  small business owner
                - Political belief:  moderate democrat
                - Religion:  Baptist
            What's your attitude towards getting COVID vaccination? 
            Attitude: probably yes.

            Example B:
            Pretend you are this person: 
                - Name:  Ava Green
                - Gender:  female
                - Age:  27 years old
                - Education:  college degree in science
                - Occupation:  stay-at-home mom
                - Political belief:  Republican
                - Religion:  Baptist
            What's your attitude towards getting COVID vaccination? 
            Attitude: probably no.

            Pretend you are this person: {profile_str_1}\n
            Choose from definitely yes, probably yes, probably no, definitely no.
            What's your attitude towards getting COVID vaccination? 
            Attitude: 
        '''
     },
]

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
encodeds_0 = tokenizer.apply_chat_template(messages_0, tokenize=False)
encodeds_1 = tokenizer.apply_chat_template(messages_1, tokenize=False)
sp = SamplingParams(
    top_p = 0.7,
    temperature = 1.5,
    max_tokens = 80
)
llm = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=1)


attitudes_0 = []
attitudes_1 = []
res_0_s = []
res_1_s = []
# simulation = Engine(1)
for i in range(30):
    # output = llm.generate(prompt_token_ids = model_inputs, sampling_params = sp)
    output = llm.generate([encodeds_0, encodeds_1], sampling_params = sp)
    res_0 = output[0].outputs[0].text
    res_0 = clean_response(res_0)
    res_0_s.append(res_0)
    attitude_0 = parse_attitude(res_0)
    attitudes_0.append(attitude_0[0])
    res_1 = output[1].outputs[0].text
    res_1 = clean_response(res_1)
    res_1_s.append(res_1)
    attitude_1 = parse_attitude(res_1)
    attitudes_1.append(attitude_1[0])

print(Counter(attitudes_0))
print(Counter(attitudes_1))

# with open("res_0.pkl", "wb") as f:
#     pickle.dump(res_0_s, f)
# with open("res_1.pkl", "wb") as f:
#     pickle.dump(res_1_s, f)







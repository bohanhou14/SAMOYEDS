
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import parse_profile
import pandas as pd
import pickle

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
state = 'Maryland'

messages = [
    {"role": "user",
     "content":
        f'''Generate a demographic profile of someone from 
            {state} that feels hesitant about COVID vaccination.
            
            Example: 
                - name: Garcia Marquez
                - gender: male
                - race: Hispanic
                - age: 45 years old
                - occupation: farm owner
                - religion: atheist
                - political belief: neutral
            
            Generate profile:
        '''
     },
#    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
profiles = []

for p in [0.7, 1]:
    for temp in [0.7, 1.0, 1.5, 2.0]:
        for i in range(10):
            generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True, top_p=p, temperature=temp)
            decoded = tokenizer.batch_decode(generated_ids)[0]
            profile = parse_profile(decoded)
            profiles.append(profile)

        with open(f'profiles/profiles-top_p={p}-temp={temp}.pkl', 'wb') as f:
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



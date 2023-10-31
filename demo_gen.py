
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
state = 'md'

messages = [
    {"role": "user",
     "content":
        f'''Generate a demographic profile of someone from 
            {state} that feels hesitant about COVID vaccination.
            
            Example: 
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

generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True, top_p=0.7, temperature=0.7)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

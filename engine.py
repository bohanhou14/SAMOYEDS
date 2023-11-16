from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import clean_response
class Engine:
    def __init__(self, max_num_agents, agents: list = None):
        if agents != None:
            # a list of agents
            self.agents = agents

        self.max_num_agents = max_num_agents
        self.conversations = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.top_p = 0.7
        self.temperature = 1.5
        # directed graph
        self.social_network = {}
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto').eval()
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    # ready to parallel run this
    def generate(self, messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=True, top_p=self.top_p, temperature=self.temperature,
                       pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids)[0]
        return decoded

    def run(self):
        return
    # def update_attitudes(self):
    #
from transformers import AutoModelForCausalLM, AutoTokenizer

class Engine:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        # a list of agents
        self.agents = []
        # directed graph
        self.social_network = {}
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto').eval()
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    # ready to parallel run this
    def generate(self, messages):


    def run(self):
        return

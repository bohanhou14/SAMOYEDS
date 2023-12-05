from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import clean_response, parse_attitude, compile_tweets
from collections import Counter
from vllm import LLM, SamplingParams
from tqdm import trange
import os
import numpy as np
import pickle
class Engine:
    def __init__(
        self, 
        agents: list = None, 
        num_gpus = 1, 
        tweets_data = None,
        news_data = None
    ):
        if agents != None:
            # a list of agents
            self.agents = agents
        self.num_agents = len(agents)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_params = SamplingParams(
            top_p = 0.7,
            temperature = 1.5
        )
        # keep track of all the conversations, shape=(num_agents X (dynamic) num_conversation_turns)
        # more turns there is, the longer messages_list will become
        self.messages_list = None
        # directed graph
        self.social_network = {}
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=num_gpus)

    # ready to parallel run this
    def generate(self, messages):
        validate_message(messages)
        model_inputs = self.tokenizer.apply_chat_template(messages, tokenize =False)
        output = self.model.generate(model_inputs, self.sampling_params)
        res = output[0].outputs[0].text
        return res
    
    def batch_generate(self, messages_list=None, max_tokens = 80):
        if messages_list != None and type(messages_list) != list:
            raise TypeError("Invalid format")
        def convert(msg):
            return self.tokenizer.apply_chat_template(msg, tokenize=False)
        if messages_list == None:
            if self.messages_list == None:
                raise RuntimeError("Messages_list not initialized yet")
            messages_list = self.messages_list
        model_inputs = [convert(msg) for msg in messages_list]
        sampling_params = self.sampling_params
        sampling_params.max_tokens = max_tokens
        output = self.model.generate(model_inputs, sampling_params)
        responses = [output[i].outputs[0].text for i in range(len(messages_list))]
        return responses

    def init_agents(self, max_iter = 30, cache_path = None, save_dir=f"./cache/default"):
        if cache_path != None:
            if os.path.exists(cache_dir):
                with open(cache_path, "rb") as f:
                    self.messages_list = pickle.load(f)
                return
            else:
                raise RuntimeError("Cache dir not exist")
        def init_impersonation(profile_str):
            return [{"role": "user",
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

                            Pretend you are this person: {profile_str}\n
                            Choose from definitely yes, probably yes, probably no, definitely no.
                            What's your attitude towards getting COVID vaccination? 
                            Attitude: 
                        '''
                    }]
        self.messages_list = []
        for agent in self.agents:
            self.messages_list.append(init_impersonation(agent.get_profile_str()))
        
        # attitudes of all agents after max_iter
        # each row corresponds to an agent's attitude X max_iter times
        all_attitudes = np.empty([self.num_agents, max_iter], dtype=object)
        # doing max_iter because llm's attitude might not be definite
        for i in trange(max_iter):
            responses = self.batch_generate(self.messages_list)
            attitudes = [parse_attitude(r)[0] for r in responses]
            all_attitudes[:, i] = attitudes
        
        for j in range(self.num_agents):
            att = all_attitudes[j]
            counter = Counter(att)
            # take the most frequent attitude
            self.agents[j].attitude = counter.most_common(1)[0][0]
        
        # update the message lists
        for k in range(self.num_agents):
            self.messages_list[k].append(
                {"role": "assistant", "content": f"Your answer: {self.agents[k].attitude}"}
            )
        if save_dir != None:
            with open(os.path.join(save_dir, f"num-agents={self.num_agents}-agent_attitudes.pkl"), "wb") as f:
                pickle.dump(self.messages_list, f)

    def feed_tweets(self, tweets: list, k=5):
        k = min(k, len(tweets))
        if type(tweets) == list:
            tweets = compile_tweets(tweets)
        prompt = {
            "role": "user",
            "content": f"You read following tweets about COVID:\n {tweets}\n What have you learned? Summarize {k} lessons you have learned: "
        }
        for k in range(self.num_agents):
            self.messages_list[k].append(prompt)
        responses = self.batch_generate(self.messages_list, max_tokens = 500)
        cleaned = [clean_response(r) for r in responses]
        return cleaned
    # TO-DO
    def validate_message(messages):
        return True
    def run(self):
        return
    # def update_attitudes(self):



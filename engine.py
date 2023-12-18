from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import clean_response, parse_attitude, compile_enumerate, parse_enumerated_items, parse_actions
from collections import Counter
from vllm import LLM, SamplingParams
from tqdm import trange
import os
from recommender import Recommender
import numpy as np
from prompts import *
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
        self.tweets_pool = []
        # directed graph
        self.social_network = {}
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=num_gpus)
        self.save_dir = f"./run_cache/default"
        self.day = 1
        self.recommender = Recommender()

    # ready to parallel run this
    
    def batch_generate(self, messages_list=None, max_tokens = 80, sampling=True):
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
        sampling_params.sampling = sampling
        output = self.model.generate(model_inputs, sampling_params)
        responses = [output[i].outputs[0].text for i in range(len(messages_list))]
        return responses

    def update_message_lists(self, new_messages):
        for k in range(self.num_agents):
            self.messages_list[k].append(
                {
                    "role": "assistant",
                    "content": new_messages[k]
                }
            )

    def save(self):
        if self.save_dir != None:
            with open(os.path.join(self.save_dir, f"num-agents={self.num_agents}-{self.stage}.pkl"), "wb") as f:
                pickle.dump(self.messages_list, f)

    def init_agents(self, max_iter = 30, cache_path = None):
        if cache_path != None:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.messages_list = pickle.load(f)
                return
            else:
                raise RuntimeError("Cache dir not exist")
        self.messages_list = []
        for agent in self.agents:
            self.messages_list.append(profile_prompt(agent.get_profile_str()))

        # greedy decoding to get the most dominant attitude
        responses = self.batch_generate(self.messages_list, sampling=False)
        attitudes = [parse_attitude(r)[0] for r in responses]

        for j in range(self.num_agents):
            self.agents[j].attitude = attitudes[j]
        
        # update the message lists
        for k in range(self.num_agents):
            self.messages_list[k].append(
                {"role": "assistant", "content": f"Your answer: {self.agents[k].attitude}"}
            )
        self.stage = f"init_agents_day={self.day}"
        self.save()

    def feed_tweets(self, k=3, num_recommendations = 10):
        profiles = [agent.get_profile_str() for agent in self.agents]
        tweets_list = self.recommender.recommend(self.tweets_pool, profiles, num_recommendations) # e.g. 500 (num_agents) * 10 (num_tweets)
        k = min(k, len(tweets_list[0]))
        tweets_list = compile_enumerate(tweets for tweets in tweets_list)
        for k in range(self.num_agents):
            self.messages_list[k].append(tweets_prompt(tweets_list[k], k))
        responses = self.batch_generate(self.messages_list, max_tokens = 500)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_tweets_day={self.day}"
        self.save()

    def feed_news_and_policies(self, news: list, policies: list = None, k=3):
        k = min(k, len(news))
        if type(news) == list:
            tweets = compile_enumerate(news)
        for k in range(self.num_agents):
            self.messages_list[k].append(news_policies_prompt(news, policies))
        responses = self.batch_generate(self.messages_list, max_tokens = 500)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_news_and_policies_day={self.day}"
        self.save()

    def prompt_reflections(self):
        self.update_message_lists(REFLECTION_PROMPT)
        responses = self.batch_generate(self.messages_list)
        cleaned = [clean_response(r) for r in responses]
        reflections = cleaned
        self.update_message_lists(reflections)
        self.stage = f"prompt_reflections_day={self.day}"
        self.save()
        return reflections
    def prompt_actions(self):
        self.update_message_lists(ACTION_PROMPT)
        responses = self.batch_generate(self.messages_list)
        cleaned = [clean_response(r) for r in responses]
        actions = parse_actions(cleaned)
        print(actions[:10])
        self.update_message_lists(actions)
        self.stage = f"prompt_actions_day={self.day}"
        self.save()
        return actions

    def poll_attitude(self):
        self.update_message_lists(ATTITUDE_PROMPT)
        responses = self.batch_generate(self.messages_list)
        cleaned = [clean_response(r) for r in responses]
        attitudes = parse_attitude(cleaned)
        for k in range(self.num_agents):
            self.messages_list[k].append(
                {"role": "assistant", "content": f"Your answer: {self.agents[k].attitude}"}
            )
        self.stage = f"poll_attitude_day={self.day}"
        self.save()
        return attitudes

    # TO-DO
    def validate_message(messages):
        return True

    def run(self):
        return
    # def update_attitudes(self):



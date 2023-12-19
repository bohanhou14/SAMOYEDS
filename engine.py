import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import clean_response, parse_attitude, compile_enumerate, parse_enumerated_items, parse_actions, counter_to_ordered_list
from collections import Counter
from vllm import LLM, SamplingParams
from tqdm import trange
import os
from tweet import Tweet
from recommender import Recommender
import numpy as np
from prompts import *
import pickle
class Engine:
    def __init__(
        self, 
        agents: list = None, 
        num_gpus = 1,
        num_days = 30
    ):
        if agents != None:
            # a list of agents
            self.agents = agents
        self.num_days = num_days
        self.num_agents = len(agents)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_params = SamplingParams(
            top_p = 0.7,
            temperature = 1.5
        )
        # keep track of all the conversations, shape=(num_agents X (dynamic) num_conversation_turns)
        # more turns there is, the longer messages_list will become
        self.messages_list = None
        with open("./data/combined_posts_texts_covid.pkl", "rb") as f:
            self.tweets_pool = pickle.load(f)
        # directed graph
        self.social_network = {}
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=num_gpus)
        self.save_dir = f"./run_cache/default"
        self.attitude_dist = []
        self.attitude_dist_4 = []
        self.day = 1
        self.recommender = Recommender()

        with open("data/news.pkl", "rb") as f:
            self.news = pickle.load(f)
            f.close()

        with open("data/policies.pkl", "rb") as f:
            self.policies = pickle.load(f)
            f.close()


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
    def add_prompt(self, new_prompts):
        # different prompt for each agent
        if type(new_prompts) == list:
            for k in range(self.num_agents):
                self.messages_list[k].append(
                    new_prompts[k]
                )
        # same prompt
        else:
            for k in range(self.num_agents):
                self.messages_list[k].append(
                    new_prompts
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
        tweets_list = self.recommender.recommend(self.tweets_pool, self.agents, num_recommendations) # e.g. 500 (num_agents) * 10 (num_tweets)
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

    def feed_news_and_policies(self, policy = None, k=3, num_news = 5):
        news = self.news[self.day-1: self.day - 1 + num_news]
        k = min(k, len(news))
        if type(news) == list:
            tweets = compile_enumerate(news)
        for k in range(self.num_agents):
            self.messages_list[k].append(news_policies_prompt(news, policy))
        responses = self.batch_generate(self.messages_list, max_tokens = 500)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_news_and_policies_day={self.day}"
        self.save()

    def prompt_reflections(self):
        # self.update_message_lists(REFLECTION_PROMPT)
        for k in range(self.num_agents):
            self.messages_list[k].append(REFLECTION_PROMPT)
        responses = self.batch_generate(self.messages_list, max_tokens=50)
        cleaned = [clean_response(r) for r in responses]
        self.stage = f"prompt_reflections_day={self.day}"
        self.update_message_lists(cleaned)
        self.save()
        return cleaned

    def prompt_actions(self):
        # self.update_message_lists(ACTION_PROMPT)
        self.add_prompt(ACTION_PROMPT)
        responses = self.batch_generate(self.messages_list)
        cleaned = [clean_response(r) for r in responses]
        # print(cleaned)
        actions = [parse_actions(c) for c in cleaned]
        self.stage = f"prompt_actions_day={self.day}"
        self.update_message_lists(actions)
        for k in range(self.num_agents):
            self.agents[k].tweets.append(text=Tweet(actions[k], time=self.day))
        self.tweets_pool.extend(actions)
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
        attitudes_counter = Counter(attitudes)
        # num_accept = attitudes_counter['definitely yes'] + attitudes_counter['probably yes']
        num_hesitancy = attitudes_counter['definitely no'] + attitudes_counter['probably no']
        hesitancy_percentage = num_hesitancy / self.num_agents
        self.attitude_dist.append(hesitancy_percentage)
        self.stage = f"poll_attitude_day={self.day}"
        self.save()
        return attitudes

    def finish_simulation(self):
        d = {
            "time": list(range(self.num_days)),
            "percentage": self.attitude_dist
        }
        json_object = json.dumps(d)
        with open("vaccine_hesitancy.json", "w") as f:
            f.write(json_object)

    def run_all_policies(self):
        for i in range(len(self.policies)):
            policy = self.policies[i]
            self.run(policy)


    def run(self, policy):
        for t in range(self.num_days):
            self.init_agents()
            self.feed_tweets()
            self.feed_news_and_policies(policy=policy)
            self.prompt_actions()
            self.prompt_reflections()
        self.finish_simulation()

    # TO-DO
    def validate_message(messages):
        return True

    # def update_attitudes(self):



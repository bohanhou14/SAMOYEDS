import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import clean_response, parse_attitude, compile_enumerate, parse_enumerated_items, parse_actions, counter_to_ordered_list, HESITANCY
from collections import Counter
from vllm import LLM, SamplingParams
from tqdm import trange, tqdm
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
        num_days = 30,
        save_dir = None
    ):
        # a list of agents
        self.agents = agents
        self.num_days = num_days
        self.agents_copy = agents
        self.num_agents = len(agents)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_params = SamplingParams(
            top_p = 0.96,
            temperature = 0.7
        )
        # keep track of all the conversations, shape=(num_agents X (dynamic) num_conversation_turns)
        # more turns there is, the longer messages_list will become
        self.messages_list = None

        # directed graph TBD
        self.social_network = {}

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
        self.model = LLM("Qwen/Qwen1.5-7B-Chat", tensor_parallel_size=num_gpus)

        self.save_dir = f"./run_cache/default" if save_dir == None else save_dir

        self.attitude_dist = []
        self.day = 1
        self.recommender = Recommender()

        with open("data/combined_posts_texts_covid.pkl", "rb") as f:
            self.tweets_pool = pickle.load(f)
            f.close()
        self.tweets_pool = [Tweet(text, time=self.day) for text in self.tweets_pool]
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
        
        def convert_mistral(msg):
            return self.tokenizer.apply_chat_template(msg, tokenize=False)
        
        def convert_qwen(msg):
            if type(msg) != list:
                msg = [msg]
            text = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            return text
        if messages_list == None:
            if self.messages_list == None:
                raise RuntimeError("Messages_list not initialized yet")
            messages_list = self.messages_list
        
        model_inputs = [convert_qwen(msg) for msg in messages_list]
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
    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.messages_list = pickle.load(f)
                f.close()

    def save(self):
        if self.save_dir != None:
            with open(os.path.join(self.save_dir, f"num-agents={self.num_agents}-{self.stage}.pkl"), "wb") as f:
                pickle.dump(self.messages_list, f)
                f.close()

    def init_agents(self, max_iter = 10, cache_path = None, openai = False):
        if cache_path != None:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.messages_list = pickle.load(f)
                return
            else:
                raise RuntimeError("Cache dir not exist")
            
        self.messages_list = []
        for i in range(self.num_agents):
            agent = self.agents[i]
            # self.messages_list.append([{"role": "system", "content": f"You are a person with this profile: {agent.get_profile_str()}"}])
            self.messages_list.append([{"role": "user", "content": f"{profile_prompt(agent.get_profile_str())}"}])

        if openai:
            responses = []
            attitudes = []
            for i in trange(len(self.messages_list)):
                attitude = ""
                attempts = 0
                while attitude == "" or attempts > max_iter:
                    response = query_openai_messages(self.messages_list[i], model = "gpt-4")
                    attitude = parse_attitude(response)[0]
                    attempts += 1
                responses.append(response)
                attitudes.append(attitude)
        else:
            # greedy decoding to get the most dominant attitude
            responses = self.batch_generate(self.messages_list, sampling=False, max_tokens=200)
        
        # update the message lists
        for j in range(self.num_agents):
            self.agents[j].attitudes.append(attitudes[j])
            self.messages_list[j].append(
                {"role": "assistant", "content": f"Attitude towards COVID vaccination: {self.agents[j].attitudes[-1]}"}
            )            
        self.stage = f"init_agents_day={self.day}"
        self.save()

    def feed_tweets(self, top_k=3, num_recommendations = 10):
        tweets_list = self.recommender.recommend(self.tweets_pool, current_day=self.day, agents=self.agents, num_recommendations=num_recommendations) # e.g. 500 (num_agents) * 10 (num_tweets)
        for k in range(self.num_agents):
            self.messages_list[k].append(tweets_prompt(tweets_list[k], top_k))
        responses = self.batch_generate(self.messages_list, max_tokens = 120)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_tweets_day={self.day}"
        self.save()

    def feed_news_and_policies(self, policy = None, top_k=3, num_news = 5):
        # print(self.news)
        news = self.news[(self.day-1): (self.day-1 + num_news)]
        top_k = min(top_k, len(news))
        if type(news) == list:
            news = compile_enumerate(news)
        for k in range(self.num_agents):
            self.messages_list[k].append(news_policies_prompt(news, policy, top_k=5))
        responses = self.batch_generate(self.messages_list, max_tokens = 120)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_news_and_policies_day={self.day}"
        self.save()

    def prompt_actions(self):
        # self.update_message_lists(ACTION_PROMPT)
        self.add_prompt(ACTION_PROMPT)
        responses = self.batch_generate(self.messages_list)
        cleaned = [clean_response(r) for r in responses]
        # print(cleaned)
        actions = [parse_actions(c) for c in cleaned]
        actions_tweets = [Tweet(a, time=self.day) for a in actions]
        self.stage = f"prompt_actions_day={self.day}"
        self.update_message_lists(actions)
        for k in range(self.num_agents):
            self.agents[k].tweets.append(actions_tweets[k])
        self.tweets_pool.extend(actions_tweets)
        self.save()
        return actions

    def poll_attitude(self):
        self.add_prompt(ATTITUDE_PROMPT)
        responses = self.batch_generate(self.messages_list)
        attitudes = [parse_attitude(r)[0] for r in responses]
        print("attitudes: ", attitudes)
        for k in range(self.num_agents):
            self.agents[k].attitudes.append(attitudes[k])
            self.messages_list[k].append(
                {"role": "assistant", "content": f"Your answer: {self.agents[k].attitudes}"}
            )
        attitudes_counter = Counter(attitudes)
        # num_accept = attitudes_counter['definitely yes'] + attitudes_counter['probably yes']
        num_hesitancy = attitudes_counter['definitely no'] + attitudes_counter['probably no']
        hesitancy_percentage = num_hesitancy / self.num_agents
        self.attitude_dist.append(hesitancy_percentage)
        self.stage = f"poll_attitude_day={self.day}"
        self.save()
        return attitudes

    def prompt_reflections(self):
        # self.update_message_lists(REFLECTION_PROMPT)
        self.add_prompt(REFLECTION_PROMPT)
        responses = self.batch_generate(self.messages_list, max_tokens=50)
        reflections = [clean_response(r) for r in responses]
        for k in range(len(reflections)):
            reflection = reflections[k]
            if (len(self.agents[k].attitudes) >= 2) and (self.agents[k].attitudes[-1] != self.agents[k].attitudes[-2]):
                self.agents[k].changes.append(reflection)
            self.agents[k].reflections.append(reflection)
        self.stage = f"prompt_reflections_day={self.day}"
        self.update_message_lists(reflections)
        self.save()
        return reflections

    def endturn_reflection(self, top_k = 5):
        self.add_prompt(ENDTURN_REFLECTION_PROMPT)
        responses = self.batch_generate(self.messages_list)
        cleaned = [clean_response(r) for r in responses]
        reasons = categorize_reasons(cleaned)
        rejection_reasons = []
        for k in range(self.num_agents):
            agent = self.agents[k]
            if agent.attitudes[-1] in HESITANCY:
                rejection_reasons.append(reasons[k])

        counter = Counter(rejection_reasons)
        rej_reasons, freqs = counter_to_ordered_list(counter)
        freqs = [(f / len(rej_reasons)) for f in freqs]
        return rej_reasons[:top_k], freqs[:top_k]

    def finish_simulation(self, id, policy, top_k=5):
        reject_reasons, reject_freqs = self.endturn_reflection(top_k)
        swing_agents = []
        for k in range(self.num_agents):
            agent = self.agents[k]
            if len(agent.changes) > top_k:
                swing_agents.append(agent)
        d = {
            "policy": policy,
            "vaccine_hesitancy_ratio": self.attitude_dist,
            "top_5_reasons_for_vaccine_hesitancy": reject_reasons,
            "top_5_reasons_for_vaccine_hesitancy_ratio": reject_freqs,
            "swing_agents": [{f"agent_{agent.name}": agent.get_json()} for agent in swing_agents]
        }
        json_object = json.dumps(d)
        with open(f"simulation_id={id}.json", "w") as f:
            f.write(json_object)

        # erase
        self.messages_list = None
        del(self.agents)

        self.agents = self.agents_copy
        self.day = 1
        self.attitude_dist = []
        del(self.tweets_pool)
        with open("data/combined_posts_texts_covid.pkl", "rb") as f:
            self.tweets_pool = pickle.load(f)

    def run(self, id, policy):
        self.init_agents(openai=True)
        for t in trange(self.num_days, desc=f"Running simulations of policy={id}"):
            self.feed_tweets()
            self.feed_news_and_policies(policy=policy)
            self.prompt_actions()
            self.poll_attitude()
            self.prompt_reflections()
            self.day += 1
        self.finish_simulation(id, policy)

    def run_all_policies(self):
        for i in range(len(self.policies)):
            policy = self.policies[i]
            self.run(i, policy)
    # TO-DO
    def validate_message(messages):
        return True

    # def update_attitudes(self):



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
from openai import OpenAI
from prompts import *
from utils import ATTITUDES
import pickle

class Engine:
    def __init__(
        self, 
        agents: list = None, 
        num_gpus = 1,
        num_days = 30,
        model_type = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        req_server = True,
        port = 8000,
        max_iter = 10,
        save_dir = None
    ):
        # a list of agents
        self.agents = agents
        self.num_days = num_days
        self.agents_copy = agents
        self.num_agents = len(agents)
        self.model_type = model_type
        self.port = port
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_params = SamplingParams(
            top_p = 0.96,
            temperature = 0.7
        )
        # keep track of all the conversations, shape=(num_agents X (dynamic) num_conversation_turns)
        # more turns there is, the longer messages_list will become
        self.messages_list = None
        self.req_server = req_server
        # directed graph TBD
        self.social_network = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        # if not inference by submitting request to servers, then we start the local model
        if not req_server:
            self.model = LLM(model_type, tensor_parallel_size=num_gpus)
        else:
            openai_api_base = f"http://0.0.0.0:{self.port}/v1"
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(
                base_url=openai_api_base,
                api_key=openai_api_key
            )
        self.save_dir = f"./run_cache/default" if save_dir == None else save_dir
        self.attitude_dist = []
        self.day = 1
        self.max_iter = max_iter
        self.recommender = Recommender(device = self.device)

        # with open("data/combined_posts_texts_covid.pkl", "rb") as f:
        #     self.tweets_pool = pickle.load(f)
        #     f.close()
        # self.tweets_pool = [Tweet(text, time=self.day) for text in self.tweets_pool]
        self.tweets_pool = []
        with open("data/news.pkl", "rb") as f:
            self.news = pickle.load(f)
            f.close()
        with open("data/policies.pkl", "rb") as f:
            self.policies = pickle.load(f)
            f.close()

    def request_generate(self, prompt, max_tokens = 80, sampling=True, top_p=None, temperature=None):
        top_p = self.sampling_params.top_p if top_p == None else top_p
        temperature = self.sampling_params.temperature if temperature == None else temperature
        completion = self.client.chat.completions.create(
            model=self.model_type,
            messages=prompt,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message.content

    # parallel inference on local model
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
                profile = self.agents[k].get_profile_str()
                self.messages_list[k].append(
                    {"role": "system", "content": f"Pretend are a person with this profile: {profile}"}
                )
                self.messages_list[k].append(
                    new_prompts[k]
                )
        # same prompt
        else:
            for k in range(self.num_agents):
                profile = self.agents[k].get_profile_str()
                self.messages_list[k].append(
                    {"role": "system", "content": f"Pretend are a person with this profile: {profile}"}
                )
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

    def init_agents(self, cache_path = None):
        if cache_path != None:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.messages_list = pickle.load(f)
                return
            else:
                raise RuntimeError("Cache dir not exist")
            
        self.messages_list = []
        responses = []; attitudes = []
        for i in range(self.num_agents):
            agent = self.agents[i]
            # self.messages_list.append([{"role": "system", "content": f"You are a person with this profile: {agent.get_profile_str()}"}])
            self.messages_list.append([{"role": "user", "content": f"{profile_prompt(agent.get_profile_str())}"}])
        if self.req_server:
            for i in trange(self.num_agents, desc="Initializing agents"):
                attitude = None; num_iter = 0
                while attitude not in ATTITUDES and num_iter < self.max_iter:
                    response = self.request_generate(self.messages_list[i], max_tokens=40)
                    attitude = parse_attitude(response)[0]
                    if attitude in ATTITUDES:
                        responses.append(response)
                        attitudes.append(attitude)
                        break
                    else:
                        num_iter += 1
                if num_iter >= self.max_iter and attitude not in ATTITUDES:
                    attitudes.append("probably no") # default assignment
        else:
            # greedy decoding to get the most dominant attitude
            responses = self.batch_generate(self.messages_list, sampling=False, max_tokens=40)
            attitudes = [parse_attitude(r)[0] for r in responses]
        print(Counter(attitudes))

        # update the message lists
        for j in range(self.num_agents):
            self.agents[j].attitudes.append(attitudes[j])
            self.messages_list[j].append(
                {"role": "assistant", "content": f"Attitude towards COVID vaccination: {self.agents[j].attitudes[-1]}"}
            )            
        self.stage = f"init_agents_day={self.day}"
        self.save()

    def feed_tweets(self, top_k=3, num_recommendations = 5):
        tweets_list = self.recommender.recommend(self.tweets_pool, current_day=self.day, agents=self.agents, num_recommendations=num_recommendations) # e.g. 500 (num_agents) * 10 (num_tweets)
        for k in range(self.num_agents):
            self.messages_list[k].append(tweets_prompt(tweets_list[k], top_k))
        responses = []
        if self.req_server:
            for i in trange(self.num_agents, desc="Feeding tweets"):
                response = self.request_generate(self.messages_list[i], max_tokens=100)
                responses.append(response)
        else:
            responses = self.batch_generate(self.messages_list, max_tokens = 100)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_tweets_day={self.day}"
        self.save()

    def feed_news_and_policies(self, policy = None, num_news = 5):
        news = self.news[(self.day-1): (self.day-1 + num_news)]
        news = [" ".join(n.split(" ")[:100]) for n in news]
        if type(news) == list:
            news = compile_enumerate(news)
        for k in range(self.num_agents):
            profile = self.agents[k].get_profile_str()
            self.messages_list[k].append(
                {
                    "role": "system",
                    "content": f"Pretend you are a person with this profile: {profile}, "
                }
            )
            self.messages_list[k].append(news_policies_prompt(news, policy, top_k=5))
        if self.req_server:
            responses = []
            for i in trange(self.num_agents, desc="Feeding news and policies"):
                response = self.request_generate(self.messages_list[i], max_tokens=120)
                responses.append(response)
        else:
            responses = self.batch_generate(self.messages_list, max_tokens = 120)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_news_and_policies_day={self.day}"
        self.save()

    def feed_news(self, num_news = 5):
        news = self.news[(self.day-1): (self.day-1 + num_news)]
        news = [" ".join(n.split(" ")[:100]) for n in news]
        # breakpoint()
        if type(news) == list:
            news_str = compile_enumerate(news)
        
        for k in range(self.num_agents):
            profile = self.agents[k].get_profile_str()
            self.messages_list[k].append(
                {
                    "role": "system",
                    "content": f"Pretend you are a person with this profile: {profile}, "
                }
            )
            self.messages_list[k].append(news_prompt(news_str))

        
        if self.req_server:
            responses = []
            for i in trange(self.num_agents, desc="Feeding news"):
                response = self.request_generate(self.messages_list[i], max_tokens=120)
                responses.append(response)
        else:
            responses = self.batch_generate(self.messages_list, max_tokens = 120)
        cleaned = [clean_response(r) for r in responses]
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned)
        self.stage = f"feed_news_day={self.day}"
        self.save()

    def prompt_actions(self):
        # self.update_message_lists(ACTION_PROMPT)
        self.add_prompt(ACTION_PROMPT)
        if self.req_server:
            responses = []
            for i in trange(self.num_agents, desc="Prompting actions"):
                response = self.request_generate(self.messages_list[i], max_tokens=100, temperature=1, top_p=1)
                responses.append(response)
        else:
            responses = self.batch_generate(self.messages_list)
        cleaned = [clean_response(r) for r in responses]
        print(cleaned)
        actions = [parse_actions(c) for c in cleaned]
        print(actions)
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
        responses = []; attitudes = []
        if self.req_server:
            for i in trange(self.num_agents, desc="Polling attitudes"):
                attitude = None; num_iter = 0
                while attitude not in ATTITUDES and num_iter < self.max_iter:
                    response = self.request_generate(self.messages_list[i], max_tokens=100)
                    attitude = parse_attitude(response)[0]
                    if attitude in ATTITUDES:
                        responses.append(response)
                        attitudes.append(attitude)
                        break
                    else:
                        num_iter += 1
                if num_iter >= self.max_iter and attitude not in ATTITUDES:
                    attitudes.append("probably no") # default assignment
        else:
            responses = self.batch_generate(self.messages_list)
            attitudes = [parse_attitude(r)[0] for r in responses]
        print(f"attitudes polled on day {self.day}: ", attitudes)
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
        if self.req_server:
            responses = []
            for i in trange(self.num_agents, desc="Prompting reflections"):
                response = self.request_generate(self.messages_list[i], max_tokens=50)
                responses.append(response)
        else:
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
        if self.req_server:
            responses = []
            for i in trange(self.num_agents, desc="Prompting endturn reflections"):
                response = self.request_generate(self.messages_list[i], max_tokens=50)
                responses.append(response)
        else:
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
        for t in trange(self.num_days, desc=f"Running simulations of policy={id}"):
            # only generated tweets? or tweet for ICL and RAG
            # label tweets with attitudes and sample consistent tweets
            # still randomize RAG
            # skip first day of tweets
            if t > 0:
                self.feed_tweets()
            # news as environmental variables
            # policy as system prompt
            self.feed_news_and_policies(policy=policy)
            # news: 1) real-world news + 2) news within the sandbox
            self.prompt_actions()
            self.poll_attitude()
            self.prompt_reflections()
            self.day += 1
        self.finish_simulation(id, policy)
    
    def warmup(self):
        self.init_agents()
        for t in trange(min(self.num_days, 2), desc="Warmup"):
            if t > 0:
                self.feed_tweets()
            # news as environmental variables
            # policy as system prompt
            self.feed_news()
            # news: 1) real-world news + 2) news within the sandbox
            self.prompt_actions()
            self.poll_attitude()
            self.prompt_reflections()
            self.day += 1

    def run_all_policies(self):
        for i in range(len(self.policies)):
            policy = self.policies[i]
            self.warmup()
            self.run(i, policy)
    # TO-DO
    def validate_message(messages):
        return True

    # def update_attitudes(self):



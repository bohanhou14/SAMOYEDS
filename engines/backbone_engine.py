# This file contains the abstract backbone engine of the simulation
# It is useful for providing a concise overview of the simulation, i.e. see the run method
# It contains general methods such as message updates, saving, loading, etc.

from transformers import AutoTokenizer
import torch
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import trange
from sandbox.prompts import system_prompt, profile_prompt
import os
import pickle
from recommenders.tweet_recommender import TweetRecommender
from recommenders.news_recommender import NewsRecommender
from sandbox.agent import Agent
from utils.utils import parse_lessons


class BackboneEngine:
    def __init__(
        self, 
        profile_str=None,
        network_str=None,
        num_days = 30,
        model_type = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        port = 7000,
        max_iter = 5,
        news_path = "data/news-k=400.pkl",
        policies_path = "data/test_3_policies.txt",
        save_dir = None, 
        seed=42
    ):
        # a list of agents
        self.num_days = num_days
        self.model_type = model_type
        self.port = port
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_params = SamplingParams(
            top_p = 0.96,
            temperature = 0.7
        )
        # keep track of all the conversations, shape=(num_agents X (dynamic) num_conversation_turns)
        # more turns there is, the longer messages_list will become
        self.context = None
        self.run_id = 0
        self.profile_str = profile_str
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.save_dir = f"./outputs" if save_dir == None else save_dir
        self.attitude_dist = []
        self.day = 1
        self.max_iter = max_iter
        self.news_path = news_path
        self.network_str = network_str
        self.policies_path = policies_path
        self.load_news()
        self.load_policies()
        self.load_agents()
        self.seed = seed
        self.tweet_recommender = TweetRecommender() 
        

    def load_network(self):
        assert self.agents != None, "Agents must be loaded before loading the network"
       
        with open(self.network_str, "rb") as f:
            self.social_network = pickle.load(f)
            f.close()
        assert len(self.agents) == len(self.social_network), "Number of agents must match the number of agents in the social network"
        for i in range(len(self.agents)):
            self.agents[i].following = self.social_network[self.agents[i].id]

    def load_agents(self):
        with open(self.profile_str, "rb") as f:
            # a list of dictionaries
            profiles = list(pickle.load(f))
        self.agents = [Agent(p) for p in profiles]
        self.num_agents = len(self.agents)
        ids = list(range(len(self.agents)))
        self.system_prompts = []
        for i in range(len(self.agents)):
            self.agents[i].id = ids[i]
            self.system_prompts.append(system_prompt(self.agents[i].get_profile_str()))
        self.load_network()

    def load_news(self):
        with open(self.news_path, "rb") as f:
            self.news = pickle.load(f)
            f.close()
        self.news_recommender = NewsRecommender() 

    def load_policies(self):
        with open(self.policies_path, "r") as f:
            self.policies = f.readlines()
            f.close
    
    def reset(self):
        self.context = []
        self.load_agents()
        self.load_news()
        self.load_policies()
        self.attitude_dist = []
        self.tweet_recommender = TweetRecommender() 
        self.day = 1

    def reset_context(self):
        self.context = [system_prompt(self.agents[i].get_profile_str()) for i in range(self.num_agents)]
        
    def add_prompt(self, new_prompts):
        self.reset_context()
        # different prompt for each agent
        if type(new_prompts) == list:
            for k in range(self.num_agents):
                # add reflections
                self.context[k].append({
                    "role": "user", 
                    "content": self.agents[k].get_reflections() + new_prompts[k]}
                )
        # same prompt
        else:
            for k in range(self.num_agents):
                self.context[k].append({
                    "role": "user", 
                    "content": self.agents[k].get_reflections() + new_prompts}
                )
    
    def add_all_lessons(self, cleaned_responses):
        for k in range(self.num_agents):
            self.agents[k].add_lessons(parse_lessons(cleaned_responses[k], day=self.day))


    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.context = pickle.load(f)
                f.close()

    def save(self, cleaned_responses):
        if self.save_dir != None:
            file_path = os.path.join(self.save_dir, f"id={self.run_id}.txt")
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write(f"run_id: {self.run_id}\n")
            print("-" * 50)
            print(f"Saving to {file_path}")
            with open(file_path, "a") as f:
                f.write(f"stage: {self.stage}\n")
                for r in cleaned_responses:
                    f.write(f"\t{r}\n")
                f.write("\n")
            print("-" * 50)

    def init_agents(self):
        pass

    def feed_tweets(self, top_k=3, num_recommendations = 5):
        pass

    def feed_news_and_policies(self, policy = None, num_news = 5):
        pass

    def feed_news(self, num_news = 5):
        pass

    def prompt_actions(self):
        pass

    def poll_attitude(self):
        pass
    
    def get_vaccines(self):
        pass

    def prompt_reflections(self):
        pass

    def endturn_reflection(self, top_k = 5):
        pass

    def finish_simulation(self, run_id, policy, top_k=5):
        pass
    
    def warmup(self, warmup_days=1):
        print("-"*50)
        print("**WARM-UP STARTED**")
        self.init_agents()
        for t in trange(warmup_days, desc="Warmup"):
            print(f"**WARM-UP DAY {t}**")
            if t > 0:
                self.feed_tweets()
            # news as environmental variables
            # policy as system prompt
            # breakpoint()
            self.feed_news()
            # breakpoint()
            # news: 1) real-world news + 2) news within the sandbox
            self.prompt_actions()
            # breakpoint()
            self.poll_attitude()
            # breakpoint()
            self.prompt_reflections()
            # breakpoint()
            self.day += 1
        print("**WARM-UP FINISHED**")
        print("-"*50)
    
    def run(self, id, policy):
        # generate a random number as the run-id
        print("-"*50)
        print(f"**Running simulations of policy={policy}**")
        print(f"**Run ID: {self.run_id}**")
        for t in trange(self.num_days, desc=f"Running simulations of policy={id}"):
            # only generated tweets? or tweet for ICL and RAG
            # label tweets with attitudes and sample consistent tweets
            # still randomize RAG
            # skip first day of tweets
            print(f"**Run DAY {t}**")
            self.feed_tweets()
            # news as environmental variables
            # policy as system prompt
            # self.feed_news_and_policies(policy=policy)
            self.feed_news()
            # news: 1) real-world news + 2) news within the sandbox
            self.prompt_actions()
            # self.get_vaccines()
            self.poll_attitude()
            self.prompt_reflections()
            self.day += 1
        self.finish_simulation(self.run_id, policy)
        print(f"**Simulation of policy={policy} finished**")
        print("-"*50)
    
    def run_all_policies(self, warmup_days=1):
        # for i in trange(len(self.policies), desc="Running all policies"):
        for i in trange(1, desc="Running all policies"):
            policy = self.policies[i]
            news_handle = self.news_path.split("/")[-1].replace(".pkl", "")
            self.run_id = f"{datetime.now().strftime('%y-%m-%d')}_{datetime.now().strftime('%H:%M')}-news={news_handle}-policy={policy[:10]}"
            self.warmup(warmup_days=warmup_days)
            self.run(i, policy)
            self.reset()

    # TO-DO
    def validate_message(messages):
        return True




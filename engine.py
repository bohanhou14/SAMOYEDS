# This file contains an abstract engine class that orchestrates the simulation
# It contains the concrete instantiation of some prompting methods and simulation flows
# However, it does not implement generate or generate_attitude methods, which are done by DataParallelEngine in multi_engine.py
# It is less on the high-level overview and more on the concrete prompt details (except generation)

from backbone_engine import BackboneEngine
import json
from utils import compile_enumerate, counter_to_ordered_list, HESITANCY
from collections import Counter
import os
from tweet import Tweet
from prompts import *
from tqdm import trange

SHORT_TOKEN_LIMIT = 50
MED_TOKEN_LIMIT = 150
LONG_TOKEN_LIMIT = 300
FULL_TOKEN_LIMIT = 1000

class Engine(BackboneEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def init_agents(self):
        self.stage = f"init_agents_day={self.day}"
        self.messages_list = [system_prompt(agent.get_profile_str()) for agent in self.agents]
        # breakpoint()
        for k in range(self.num_agents):
            self.messages_list[k].append(
                profile_prompt(self.agents[k].get_profile_str())
            )
        # breakpoint()
        attitudes = self.generate_attitude(max_tokens=SHORT_TOKEN_LIMIT)
        # update the message lists
        for j in range(self.num_agents):
            self.agents[j].attitudes.append(attitudes[j])
            self.messages_list[j].append(
                {"role": "assistant", "content": f"Attitude towards FD vaccination: {self.agents[j].attitudes[-1]}"}
            )          
        self.update_attitude_dist(attitudes)  
        self.save()

    def generate_attitude(self, max_tokens):
        pass
    def generate(self, max_tokens):
        pass
    def generate_actions(self, max_tokens):
        pass

    def feed_tweets(self, top_k=3, num_recommendations = 5):
        self.stage = f"feed_tweets_day={self.day}"
        tweets_list = self.recommender.recommend(self.tweets_pool, current_day=self.day, agents=self.agents, num_recommendations=num_recommendations) # e.g. 500 (num_agents) * 10 (num_tweets)
        for k in range(self.num_agents):
            self.messages_list[k].append(tweets_prompt(tweets_list[k], top_k))
        self.save()
        self.stage = f"write_tweets_lesson_day={self.day}"
        cleaned_responses = self.generate(max_tokens=MED_TOKEN_LIMIT)
        self.update_message_lists(cleaned_responses)
        self.save()
    
    def feed_news_data(self, num_news):
        # return []
        news = self.news[(self.day-1): (self.day-1 + num_news)]
        news = [" ".join(n.split(" ")[:200]) for n in news]
        news = [n.replace("\n", " ") for n in news]
        if type(news) == list:
            news = compile_enumerate(news)
        return news
    
    def feed_news_and_policies(self, policy = None, num_news = 10):
        self.stage = f"feed_news_and_policies_day={self.day}"
        news = self.feed_news_data(num_news)
        for k in range(self.num_agents):
            self.messages_list[k].append(news_policies_prompt(news, policy, top_k=5))
        
        cleaned_responses = self.generate(max_tokens=MED_TOKEN_LIMIT)
        # lessons = [parse_enumerated_items(c) for c in cleaned]
        self.update_message_lists(cleaned_responses)
        self.save()
    
    def feed_news(self, num_news = 5):
        self.stage = f"feed_news_day={self.day}"
        news = self.feed_news_data(num_news)
        for k in range(self.num_agents):
            profile = self.agents[k].get_profile_str()
            self.messages_list[k].append(news_prompt(news))

        cleaned_responses = self.generate(max_tokens=MED_TOKEN_LIMIT)
        self.update_message_lists(cleaned_responses)
        self.save()

    def prompt_actions(self):
        self.stage = f"prompt_actions_day={self.day}"
        self.add_prompt(ACTION_PROMPT)
        actions = self.generate(max_tokens=MED_TOKEN_LIMIT)
        actions_tweets = [Tweet(text=actions[i], time=self.day, author_id=i) for i in range(len(actions))]
        self.update_message_lists(actions)
        for k in range(self.num_agents):
            self.agents[k].tweets.append(actions_tweets[k])
        self.tweets_pool.extend(actions_tweets)
        self.save()
        return actions
    
    def update_attitude_dist(self, attitudes):
        attitudes_counter = Counter(attitudes)
        num_hesitancy = attitudes_counter['definitely no'] + attitudes_counter['probably no']
        hesitancy_percentage = num_hesitancy / self.num_agents
        with open(f"attitude_dist-{self.stage}-{self.run_id}.txt", "a") as f:
            f.write(f"day={self.day}, hesitancy_percentage={hesitancy_percentage}\n")
            f.write(f"attitudes={attitudes_counter}\n")
            f.close()
        self.attitude_dist.append(hesitancy_percentage)

    def poll_attitude(self):
        self.add_prompt(ATTITUDE_PROMPT)
        self.stage = f"poll_attitude_day={self.day}"
        attitudes = self.generate_attitude(max_tokens=SHORT_TOKEN_LIMIT)
        for k in range(self.num_agents):
            self.agents[k].attitudes.append(attitudes[k])
            self.messages_list[k].append(
                {"role": "assistant", "content": f"Your answer: {self.agents[k].attitudes[-1]}"}
            )
        self.update_attitude_dist(attitudes)
        self.save()
        return attitudes
    
    def get_vaccines(self):
        self.add_prompt(VACCINE_PROMPT)
        self.stage = f"get_vaccines_day={self.day}"
        responses = []
        for i in trange(self.num_agents, desc="Getting vaccines"):
            response = None
            while response == None:
                if self.agents[i].vaccine:
                    responses.append("I have already gotten the vaccine.")
                else:
                    response = self.request_generate(self.messages_list[i], max_tokens=MED_TOKEN_LIMIT)
                    # print(response)
                    response = str(parse_yes_or_no(response))
            responses.append(response)
            self.agents[i].vaccine = response
        self.update_message_lists(responses)
        self.save()
        return responses
    
    def prompt_reflections(self):
        self.add_prompt(REFLECTION_PROMPT)
        self.stage = f"prompt_reflections_day={self.day}"
        reflections = self.generate(max_tokens=MED_TOKEN_LIMIT)
        for k in range(len(reflections)):
            reflection = reflections[k]
            if (len(self.agents[k].attitudes) >= 2) and (self.agents[k].attitudes[-1] != self.agents[k].attitudes[-2]):
                self.agents[k].changes.append(reflection)
            self.agents[k].reflections.append(reflection)
        
        self.update_message_lists(reflections)
        self.save()
        return reflections
    
    def endturn_reflection(self, top_k = 5):
        self.stage = f"endturn_reflection_day={self.day}"
        self.add_prompt(ENDTURN_REFLECTION_PROMPT)
        reasons = []
        all_cleaned = self.generate(max_tokens=MED_TOKEN_LIMIT)
        prompts = [get_categorization_prompt(cleaned) for cleaned in all_cleaned]
        self.add_prompt(prompts)
        self.stage = f"analyze_reasons_day={self.day}"
        reasons = self.generate(max_tokens=MED_TOKEN_LIMIT)
        
        rejection_reasons = []
        for k in range(self.num_agents):
            agent = self.agents[k]
            if agent.attitudes[-1] in HESITANCY:
                rejection_reasons.append(reasons[k])

        counter = Counter(rejection_reasons)
        rej_reasons, freqs = counter_to_ordered_list(counter)
        freqs = [(f / len(rej_reasons)) for f in freqs]
        return rej_reasons[:top_k], freqs[:top_k]
    
    def finish_simulation(self, run_id, policy, top_k=5):
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
        path = os.path.join(self.save_dir, f"simulation_id={run_id}.json")
        with open(path, "w") as f:
            f.write(json_object)
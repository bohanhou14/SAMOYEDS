# This file contains an abstract engine class that orchestrates the simulation
# It contains the concrete instantiation of some prompting methods and simulation flows
# However, it does not implement generate or generate_attitude methods, which are done by DataParallelEngine in multi_engine.py
# It is less on the high-level overview and more on the concrete prompt details (except generation)

from engines.backbone_engine import BackboneEngine
import json
from utils.utils import compile_enumerate, counter_to_ordered_list, HESITANCY, parse_lessons
from collections import Counter
import os
from sandbox.tweet import Tweet
from sandbox.prompts import *
from tqdm import trange

SHORT_TOKEN_LIMIT = 50
TWEET_TOKEN_LIMIT = 100
MED_TOKEN_LIMIT = 150
LONG_TOKEN_LIMIT = 250
FULL_TOKEN_LIMIT = 1000

class Engine(BackboneEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)      

    def init_agents(self):
        self.stage = f"init_agents_day={self.day}"
        # breakpoint()
        profile_prompts = [profile_prompt(self.agents[i].get_profile_str()) for i in range(self.num_agents)]
        self.add_prompt(profile_prompts)
        # breakpoint()
        attitudes = self.generate_attitude(max_tokens=MED_TOKEN_LIMIT)
        # update the message lists
        for j in range(self.num_agents):
            self.agents[j].attitudes.append(attitudes[j])      
        self.update_attitude_dist(attitudes)  
        self.save(attitudes)

    def generate_attitude(self, max_tokens):
        pass
    def generate(self, max_tokens):
        pass
    def generate_actions(self, max_tokens):
        pass
    
    def feed_news_data(self, num_news=5):
        search_space = num_news * num_news
        news_data = self.news[self.day * search_space: (self.day + 1) * search_space]
        recommendations = self.news_recommender.recommend(agents=self.agents, num_recommendations=num_news, news_data=news_data)
        all_news = []
        purities = []
        stances = []
        similarities = []
        for k in range(self.num_agents):
            # breakpoint()
            news_text, news_stance, news_sim = recommendations[k]
            news = compile_enumerate(news_text)
            binary_stance = [1 if s == "positive" else 0 for s in news_stance]
            purity = sum(binary_stance) / len(binary_stance) if sum(binary_stance) > len(binary_stance) / 2 else 1 - sum(binary_stance) / len(binary_stance)
            purities.append(purity)
            stances.append(sum(binary_stance) / num_news)
            similarities.append(sum(news_sim) / num_news)
            all_news.append(news)

        print(f"Average Purity: {sum(purities) / len(purities)}")
        print(f"Average Stance: {sum(stances) / len(stances)}")
        print(f"Average Similarity: {sum(similarities) / len(similarities)}")
        return all_news
    
    def feed_news_and_policies(self, policy = None, num_news = 5):
        self.stage = f"feed_news_and_policies_day={self.day}"
        news = self.feed_news_data(num_news)
        prompts = [news_policies_prompt(news[k], policy, k=num_news) for k in range(self.num_agents)]
        self.add_prompt(prompts)
        cleaned_responses = self.generate(max_tokens=LONG_TOKEN_LIMIT)
        self.add_all_lessons(cleaned_responses)
        self.save(cleaned_responses)
    
    def feed_news(self, num_news = 5):
        self.stage = f"feed_news_day={self.day}"
        news = self.feed_news_data(num_news)
        prompts = [news_prompt(news[k], k=num_news) for k in range(self.num_agents)]
        self.add_prompt(prompts)
        cleaned_responses = self.generate(max_tokens=LONG_TOKEN_LIMIT)
        lessons = [parse_lessons(c, day=self.day) for c in cleaned_responses]
        for k in range(self.num_agents):
            self.agents[k].add_lessons(lessons[k])
        self.save(cleaned_responses)
    
    def feed_tweets(self, top_k=3, num_recommendations = 5):
        self.stage = f"feed_tweets_day={self.day}"
        recommendations = self.tweet_recommender.recommend(agents=self.agents, num_recommendations=num_recommendations) # e.g. 500 (num_agents) * 10 (num_tweets)
        print("Recommendations generated")
        prompts = [tweets_prompt([r[1] for r in recommendations if r[0]==k], top_k) for k in range(self.num_agents)]
        print(f"Prompts generated, example: {prompts[0]}")
        self.add_prompt(prompts)
        self.stage = f"write_tweets_lesson_day={self.day}"
        cleaned_responses = self.generate(max_tokens=MED_TOKEN_LIMIT)
        self.add_all_lessons(cleaned_responses)
        self.save(cleaned_responses)

    def prompt_actions(self):
        self.stage = f"prompt_actions_day={self.day}"
        self.add_prompt(ACTION_PROMPT)
        breakpoint()
        actions = self.generate(max_tokens=TWEET_TOKEN_LIMIT)
        actions_tweets = [Tweet(text=actions[i], time=self.day, author_id=i) for i in range(len(actions))]
        for k in range(self.num_agents):
            self.agents[k].tweets.append(actions_tweets[k])
        self.save(actions)
        return actions
    
    def update_attitude_dist(self, attitudes):
        attitudes_counter = Counter(attitudes)
        num_hesitancy = attitudes_counter['definitely no'] + attitudes_counter['probably no']
        hesitancy_percentage = num_hesitancy / self.num_agents
        with open(os.path.join(self.save_dir,f"attitude_dist-{self.stage}-{self.run_id}.txt"), "a") as f:
            f.write(f"day={self.day}, hesitancy_percentage={hesitancy_percentage}\n")
            f.write(f"attitudes={attitudes_counter}\n")
            f.close()
        self.attitude_dist.append(hesitancy_percentage)

    def poll_attitude(self):
        self.add_prompt(ATTITUDE_PROMPT)
        self.stage = f"poll_attitude_day={self.day}"
        attitudes = self.generate_attitude(max_tokens=MED_TOKEN_LIMIT)
        for k in range(self.num_agents):
            self.agents[k].attitudes.append(attitudes[k])
        self.update_attitude_dist(attitudes)
        self.save(attitudes)
    
    def prompt_reflections(self):
        self.add_prompt(REFLECTION_PROMPT)
        self.stage = f"prompt_reflections_day={self.day}"
        reflections = self.generate(max_tokens=MED_TOKEN_LIMIT)
        for k in range(len(reflections)):
            reflection = reflections[k]
            if (len(self.agents[k].attitudes) >= 2) and (self.agents[k].attitudes[-1] != self.agents[k].attitudes[-2]):
                self.agents[k].changes.append(parse_lessons(reflection, day=self.day))
        self.add_all_lessons(reflections)
        self.save(reflections)
        return reflections
    
    def endturn_reflection(self, top_k = 5):
        self.stage = f"endturn_reflection_day={self.day}"
        self.add_prompt(REFLECTION_PROMPT)
        reasons = []
        all_cleaned = self.generate(max_tokens=MED_TOKEN_LIMIT)
        prompts = [get_categorization_prompt(cleaned) for cleaned in all_cleaned]
        self.add_prompt(prompts)
        self.stage = f"analyze_reasons_day={self.day}"
        reasons = self.generate(max_tokens=LONG_TOKEN_LIMIT)
        
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
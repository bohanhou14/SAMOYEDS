from engine import Engine
import pickle
from agent import Agent
with open('profiles/profiles-state-United States=attitude="probably no or definitely no"-agent_num=10-top_p=0.96-temp=1.0.pkl', "rb") as f:
    # a list of dictionaries
    profiles = list(pickle.load(f))

agents = []
for p in profiles:
    agents.append(Agent(profile=p))
engine = Engine(agents = agents, num_gpus=1, num_days=3, save_dir="run_cache/debug/")
engine.run_all_policies()
# engine.init_agents(openai=True)

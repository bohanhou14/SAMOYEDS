from engine import Engine
import pickle
from agent import Agent
with open("profiles/profiles-state-United States=attitude=probably_no_or_definitely_no-agent_num=10-top_p=0.7-temp=1.5.pkl", "rb") as f:
    # a list of dictionaries
    profiles = list(pickle.load(f))

agents = []
for p in profiles:
    agents.append(Agent(profile=p))
engine = Engine(agents = agents, num_gpus=1, num_days=3)
# engine.run_all_policies()
engine.init_agents()

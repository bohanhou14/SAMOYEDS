from engine import Engine
import pickle
from agent import Agent
with open("profiles/profiles-agent_num=10-top_p=0.7-temp=2.0.pkl", "rb") as f:
    # a list of dictionaries
    profiles = list(pickle.load(f))

profile_0 = profiles[0]
profile_1 = profiles[1]
profile_2 = profiles[2]

agent_0 = Agent(profile=profile_0)
agent_1 = Agent(profile=profile_1)
agent_2 = Agent(profile=profile_2)

agents = [agent_0, agent_1, agent_2]
engine = Engine(agents = agents)
engine.init_agents()

for a in engine.agents:
    print(a.attitude)
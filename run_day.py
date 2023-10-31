import pickle
from agent import Agent
from engine import Engine
from utils import read_profile

with open("profiles/profiles-agent_num=10-top_p=0.7-temp=2.0.pkl", "rb") as f:
    # a list of dictionaries
    profiles = pickle.load(f)

profile = profiles[0]
agent = Agent(profile)
profile_str = agent.get_profile_str()
simulation = Engine(1)

messages = [
    {
        "role": "user",
        "content": f'''Pretend you are a person with following characteristics: \n{profile_str}\n
                       {agent.solicit_attitude()}
                    '''
    }
]

new_messages = simulation.generate(messages)
print(new_messages)





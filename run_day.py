import pickle
from agent import Agent
from engine import Engine
from utils import clean_response

with open("profiles/profiles-agent_num=10-top_p=0.7-temp=2.0.pkl", "rb") as f:
    # a list of dictionaries
    profiles = list(pickle.load(f))

profile = profiles[0]
agent = Agent(profile=profile)
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

new_messages = simulation.generate(messages)[0]['content']
response = clean_response(new_messages)
print(response)





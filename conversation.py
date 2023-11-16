from agent import Agent
from utils import ATTITUDES, parse_attitude, clean_response

class Conversation:
    def __init__(self, profile):
        self.messages = []
        self.init_agent(profile)

    def append_inst(self, text: str):
        msg = {"role": "user", "content": clean_response(text)}
        self.messages.append(msg)

    def append_response(self, text: str):
        msg = {"role": "assistant", "content": text}
        self.messages.append(msg)

    def solicit_attitude(self):
        prompt = f'''
            Considering what you learned and your demographics,
            if a vaccine to prevent COVID-19 were offered to you today, would you choose to get vaccinated?

            Choose from:
            - {ATTITUDES[0]}
            - {ATTITUDES[1]}
            - {ATTITUDES[2]}
            - {ATTITUDES[3]}

            Example:
                Your attitude: No, probably

            Your attitude: 
        '''
        return prompt

    def run_a_day(self, engine):
        # update attitude
        self.converse(engine)
        self.agent.attitude = parse_attitude(self.get_response())

    def get_response(self):
        return self.messages[-1]['content']

    def converse(self, engine):
        decoded = engine.generate(self.messages)
        self.append_response(decoded)

    def feed_tweets(self, tweets: list):
        tweet_str = "\n"
        # start from 1
        for i in range(1, len(tweets), 1):
            tweet = tweets[0]
            tweet_str += f"{i}. {tweet}\n"
        prompt = f'''
            You read the following tweets of the day: {tweet_str}
        '''
        self.append_inst(prompt)

    # init agent with profile and solicit initial attitude
    def init_agent(self, profile: dict):
        self.agent = Agent(profile=profile)
        profile_str = self.agent.get_profile_str()
        init_prompt = f'''Pretend you are a person with following characteristics: \n{profile_str}\n
                                       {self.solicit_attitude()}
                       '''
        self.append_inst(init_prompt)






from sandbox.tweet import Tweet
import hashlib
from utils.utils import compile_enumerate
class Agent:
    def __init__(self, profile):
        self.gender = profile['Gender']
        self.age = profile['Age']
        self.occupation = profile['Occupation']
        self.education = profile['Education']
        self.pb = profile['Political belief']
        self.religion = profile['Religion']
        self.attitudes = []
        self.changes = []
        self.reflections = []
        self.tweets = []
        self.vaccine = None
        self.following = []
    
    def custom_init(self, name, gender, age, occupation, education, pb, religion):
        self.gender = gender
        self.age = age
        self.occupation = occupation
        self.education = education
        self.pb = pb
        self.religion = religion

    def update_tweets(self, tweet_text, tweet_time):
        self.tweets.append(Tweet(tweet_text, tweet_time))
    
    def get_all_tweets(self):
        return self.tweets
    
    def get_all_tweets_str(self):
        return compile_enumerate([tweet.text for tweet in self.tweets])
    
    def get_most_recent_tweets(self):
        if len(self.tweets) == 0:
            print("No tweets found. Probably the agent has not tweeted yet.")
            return None
        return self.tweets[-1]

    def get_profile_str(self):
        profile_str = f'''Gender: {self.gender}\tAge: {self.age}\tEducation: {self.education}\tOccupation: {self.occupation}\tPolitical belief: {self.pb}\tReligion: {self.religion}'''
        # profile_str = f'''Gender: {self.gender}\tAge: {self.age}\tEducation: {self.education}\tOccupation: {self.occupation}'''
        
        if len(self.attitudes) > 0:
            profile_str += f"\tAttitude towards COVID Vaccination: {self.attitudes[-1]}"
        if self.vaccine != None:
            profile_str += f"\tVaccinated or not: {self.vaccine}"
        return profile_str

    def get_json(self):
        profile_json = {
            "Gender": self.gender,
            "Age": self.age,
            "Education": self.education,
            "Occupation": self.occupation,
            "Political belief": self.pb,
            "Religion": self.religion,
            "Reasons for Changing": self.changes
        }

        return profile_json




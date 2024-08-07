from sandbox.tweet import Tweet
import hashlib
from queue import PriorityQueue 
from utils.utils import compile_enumerate
class Agent:
    def __init__(self, profile, k=10):
        self.gender = profile['Gender']
        self.age = profile['Age']
        self.occupation = profile['Occupation']
        self.education = profile['Education']
        self.pb = profile['Political belief']
        self.religion = profile['Religion']
        self.attitudes = []
        self.max_reflections = k
        self.changes = []
        self.lessons = [] # a queue of triples (reflection, time, importance)
        self.reflections = [] # the top k reflections with highest scores (lesson, score)
        self.tweets = []
        self.vaccine = None
        self.following = {} # should be a list of tuple (id, weight)
    
    def custom_init(self, gender, age, occupation, education, pb, religion):
        self.gender = gender
        self.age = age
        self.occupation = occupation
        self.education = education
        self.pb = pb
        self.religion = religion
    
    def add_lessons(self, lessons):
        self.reflections.extend(lessons)

    def retrieve_reflections(self, current_time):
        reflections = [(lesson.text, lesson.score(current_time)) for lesson in self.lessons]
        reflections.sort(key=lambda x: x[1], reverse=True)
        self.reflections = reflections[:self.max_reflections]
        return reflections[:self.max_reflections]
    
    def get_reflections(self):
        if len(self.reflections) == 0:
            return ""
        ret_str = "Below are the most influential lessons to your opinions, shown in ascending order:\n"
        ret_str += compile_enumerate([reflection[0] for reflection in self.reflections[::-1]])
        ret_str += "\n Please consider these lessons carefully when you make your decisions.\n"
        return ret_str

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




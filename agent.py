from tweet import Tweet
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
    def custom_init(self, name, gender, age, occupation, education, pb, religion):
        self.gender = gender
        self.age = age
        self.occupation = occupation
        self.education = education
        self.pb = pb
        self.religion = religion
    def update_tweets(self, tweet_text, tweet_time):
        self.tweets.append(Tweet(tweet_text, tweet_time))

    def get_profile_str(self):
        profile_str = f'''
            - Gender: {self.gender}
            - Age: {self.age}
            - Education: {self.education}
            - Occupation: {self.occupation}
            - Political belief: {self.pb}
            - Religion: {self.religion}
        '''
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




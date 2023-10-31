
class Agent:
    def __init__(self, profile: dict):
        self.name = profile['Name']
        self.gender = profile['Gender']
        self.age = profile['Age']
        self.occupation = profile['Occupation']
        self.education = profile['Education']
        self.pb = profile['Political belief']
        self.religion = profile['Religion']

    def __init__(self, name, gender, age, occupation, education, pb, religion):
        self.name = name
        self.gender = gender
        self.age = age
        self.occupation = occupation
        self.education = education
        self.pb = pb
        self.religion = religion

    def get_profile_str(self):
        profile_str = f'''
            - Name: {self.name}
            - Gender: {self.gender}
            - Age: {self.age}
            - Education: {self.education}
            - Occupation: {self.occupation}
            - Political belief: {self.pb}
            - Religion: {self.religion}
        '''
        return profile_str

    def express_attitude(self):
        prompt = '''
            Considering what you learned and your demographics,
            if a vaccine to prevent COVID-19 were offered to you today, would you choose to get vaccinated?
            
            Choose from:
            - Yes, definitely
            - Yes, probably,
            - No, probably,
            - No, definitely
            
            Example: 
                Attitude: Yes, definitely
        '''




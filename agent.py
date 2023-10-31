
class Agent:
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



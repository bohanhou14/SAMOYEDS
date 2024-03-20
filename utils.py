import numpy as np
API_KEY = '7fa6fe05da148'
import datetime
import os
import re
import nltk

# General methods
# clean the instructions in the response

def counter_to_ordered_list(counter):
    ordered_keys = []
    ordered_values = []
    for key in counter.keys():
        ordered_keys.append(key)
        ordered_values.append(counter[key])
    return ordered_keys, ordered_values
    
def clean_response(response):
    x = response.strip()
    x = re.sub('\[INST\]([\s\S]*)\[/INST\]', '', response, flags=re.DOTALL)
    x = x.replace("<s>", "")
    x = x.replace("</s>", "")
    return x

def compile_enumerate(tweets: list):
    res_str = ""
    for i in range(len(tweets)):
        res_str += f"{i+1}: {tweets[i]}\n"
    return res_str


def get_data_path(state, sig, start_date, end_date):
    assert type(start_date) == datetime.date
    assert type(end_date) == datetime.date
    file_path = os.path.join("data", f"{state}-{sig}-from-{start_date}-to-{end_date}.pkl")
    return file_path

# Fields for querying data
SIGNALS = np.array(["smoothed_whesitancy_reason_sideeffects",
                    "smoothed_whesitancy_reason_allergic",
                    "smoothed_whesitancy_reason_ineffective",
                    "smoothed_whesitancy_reason_unnecessary",
                    "smoothed_whesitancy_reason_dislike_vaccines",
                    "smoothed_whesitancy_reason_not_recommended",
                    "smoothed_whesitancy_reason_wait_safety",
                    "smoothed_whesitancy_reason_low_priority",
                    "smoothed_whesitancy_reason_cost",
                    "smoothed_whesitancy_reason_distrust_vaccines",
                    "smoothed_whesitancy_reason_distrust_gov",
                    "smoothed_whesitancy_reason_health_condition",
                    "smoothed_whesitancy_reason_pregnant",
                    "smoothed_whesitancy_reason_religious",
                    "smoothed_whesitancy_reason_other"])

SHORT_SIGNALS = np.array(["sideeffects",
                          "allergic",
                          "ineffective",
                          "unnecessary",
                          "dislike_vaccines",
                          "not_recommended",
                          "wait_safety",
                          "low_priority",
                          "cost",
                          "distrust_vaccines",
                          "distrust_gov",
                          "health_condition",
                          "pregnant",
                          "religious",
                          "other"])


# Fields and methods for profile information
PROFILE_ATTRIBUTES_LOWER = [
                            "gender:",
                            "race:",
                            "age:",
                            "occupation:",
                            "education:",
                            "religion:",
                            "political belief:"]

PROFILE_ATTRIBUTES = [
                      'Gender',
                      'Race',
                      'Age',
                      'Occupation',
                      'Education',
                      'Religion',
                      'Political belief']

ATTRIBUTES_MAP = {
    PROFILE_ATTRIBUTES_LOWER[0]: PROFILE_ATTRIBUTES[0],
    PROFILE_ATTRIBUTES_LOWER[1]: PROFILE_ATTRIBUTES[1],
    PROFILE_ATTRIBUTES_LOWER[2]: PROFILE_ATTRIBUTES[2],
    PROFILE_ATTRIBUTES_LOWER[3]: PROFILE_ATTRIBUTES[3],
    PROFILE_ATTRIBUTES_LOWER[4]: PROFILE_ATTRIBUTES[4],
    PROFILE_ATTRIBUTES_LOWER[5]: PROFILE_ATTRIBUTES[5],
    PROFILE_ATTRIBUTES_LOWER[6]: PROFILE_ATTRIBUTES[6]
}

def parse_profile(text):
    def find_attribute(line):
        for att in PROFILE_ATTRIBUTES_LOWER:
            # print(f"line: {line}, att: {att}")
            idx = line.find(att)
            # print(idx)
            if idx != -1:
                return att, idx
        return "", -1
    #
    profile = {}
    # remove the instruction
    x = clean_response(text)
    x = x.split("\n")
    for l in x:
        l = l.strip()
        # if finds the first is a bullet
        if len(l) > 0:
            # l[0] is the bullet, l[1] might be space
            # there are also instances where there are no bullets
            att, idx = find_attribute(l.lower())
            # if no attributes found, next iter
            if att != "":
                attribute = ATTRIBUTES_MAP[att]
            else: continue
            # discount the first bullet and lead space
            profile[attribute] = l[idx + len(att): ]
    return profile
#
# read a profile and return a string describing the profile
def read_profile(profile):
    gender = profile['Gender']
    religion = profile['Religion']
    pb = profile['Political belief']
    age = profile['Age']
    oc = profile['Occupation']
    ed = profile['Education']
    str = f'''
        - Gender: {gender}
        - Age: {age}
        - Education: {ed}
        - Occupation: {oc}
        - Political belief: {pb}
        - Religion: {religion}
    '''
    return str


# Fields and methods for attitudes
ATTITUDES = [
    "definitely yes",
    "probably yes",
    "probably no",
    "definitely no"
]

HESITANCY = [
    "probably no",
    "definitely no"
]

REASONS = ["sideeffects", "allergic", "ineffective",
           "unnecessary", "dislike_vaccines_generally",
           "dislike_vaccines", "not_recommended",
           "wait_safety", "low_priority", "cost",
           "distrust_vaccines", "distrust_gov",
           "health_condition", "pregnant",
           "religious", "other"]
# parse attitude based on response
def parse_attitude(response):
    x = clean_response(response)
    for att in ATTITUDES:
        idx = x.find(att)
        if idx != -1:
            return att, idx
        # if here, idx == -1
        # check if att did not get extractd because it is lower
        idx = x.lower().find(att.lower())
        if idx != -1:
            return att, idx

    # if here, then no attitude, must be something wrong
    print(f"No attitude found:\n {x}\n\n")
    return "", -1

def parse_reasons(response):
    x = clean_response(response)
    for reason in REASONS:
        idx = x.find(reason)
        if idx != -1:
            return reason, idx
        # if here, idx == -1
        # check if att did not get extractd because it is lower
        idx = x.lower().find(reason)
        if idx != -1:
            return reason, idx

    # if here, then no attitude, must be something wrong
    return "", -1

def parse_enumerated_items(text):
    # Regex pattern to match enumerated items
    pattern = r"\d+\.\s*(.+)"
    # Find all matches
    matches = re.findall(pattern, text)
    return matches

def parse_actions(text):
    output = text.split('*', 1)[1] if '*' in text else ''
    return output.strip()


if __name__ == '__main__':
    text = '''
        <s> [INST] Generate a demographic profile of someone from 
            md that feels hesitant about COVID vaccination.
            
            Example: 
                - Gender: male
                - Race: Hispanic
                - Age: 45 years old
                - Occupation: farm owner
                - Religion: atheist
                - Political belief: neutral
            
            Generate profile:
         [/INST] Based on your request, here is a demographic profile of someone from MD who feels hesitant about COVID vaccination:

            * Gender: Female
            * Race: African American
            * Age: 35 years old
            * Occupation: Small business owner
            * Education: Bachelor's degree in Business Administration
            * Religion: Christian
            * Political belief: Right-leaning
            
            This person may have concerns about the safety and effectiveness of the COVID-19 vaccines, as well as questions about potential side effects. They may also believe that the vaccines are being rushed through emergency approval processes and not thoroughly tested. Additionally, they may be concerned about the potential for the vaccines to impact their fertility or cause long-term health effects.
            
            Overall, this individual may be hesitant to get vaccinated due to a combination of these and other fears or concerns, despite the overwhelming evidence supporting the safety and effectiveness of COVID-19 vaccines.</s>
    '''
    profile = parse_profile(text)
    print(profile)



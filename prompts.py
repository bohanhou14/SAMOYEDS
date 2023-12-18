ATTITUDE_PROMPT = {
            "role": "user",
            "content": "Based on the lessons you learned and your previous attitude towards COVID vaccinations, choose your current attitude towards COVID vaccinations from {definitely no, probably no, probably yes, and definitely yes}."
        }
def profile_prompt(profile_str):
    return [{"role": "user",
             "content":
                 f'''
                    Example A:
                    Pretend you are this person: 
                        - Name:  Karen Williams
                        - Gender:  female
                        - Age:  50 years old
                        - Education:  College graduate
                        - Occupation:  small business owner
                        - Political belief:  moderate democrat
                        - Religion:  Baptist
                    What's your attitude towards getting COVID vaccination? 
                    Attitude: probably yes.

                    Example B:
                    Pretend you are this person: 
                        - Name:  Ava Green
                        - Gender:  female
                        - Age:  27 years old
                        - Education:  college degree in science
                        - Occupation:  stay-at-home mom
                        - Political belief:  Republican
                        - Religion:  Baptist
                    What's your attitude towards getting COVID vaccination? 
                    Attitude: probably no.

                    Pretend you are this person: {profile_str}\n
                    Choose from definitely yes, probably yes, probably no, definitely no.
                    What's your attitude towards getting COVID vaccination? 
                    Attitude: 
                '''
             }]
def news_policies_prompt(news, policies = None):
    prompt = {
        "role": "user",
        "content": f"You read following news today about COVID:\n {news}\n "
    }
    if policies != None:
        policy_prompt = f"The government has also issued the following policies:\n {policies}\n"
        prompt['content'] += policy_prompt
    question = f"What have you learned? Summarize {k} lessons you have learned: "
    prompt['content'] += question

    return prompt

def tweets_prompt(tweets, k=5):
    TWEETS_PROMPT = {
        "role": "user",
        "content": f"You read following tweets about COVID:\n {tweets}\n What have you learned? Summarize {k} lessons you have learned: "
    }
    return TWEETS_PROMPT

REFLECTION_PROMPT = {
            "role": "user",
            "content": '''Based on your background and the lessons you have learned, 
                reflect upon the most significant reasons that cause your attitude towards COVID vaccination to change or unchange: '''
        }

ACTION_PROMPT = {
            "role": "user",
            "content": '''Based on your attitude towards COVID vaccinations and your background, what actions do you want to take to express your feelings? 
                You can choose from {make a new tweet, comment on a tweet or news you read, do nothing}'''
        }




import openai
import backoff
from utils import parse_reasons

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
            "content": '''Based on your attitude towards COVID vaccinations and your background, make a new tweet that responds to an old tweet or start an entirely new tweet on COVID vaccination:
                
                Example A:
                * COVID vaccination is a debated topic. I hope that people can have more awareness and critical thinking and make their informed decisions based on facts and evidences than emotions.
                
                Example B:
                * I'd like to respond to this tweet I read today. I feel like this tweet is trying to gaslight people into believing that the government is harming people.

                Write a tweet about COVID vaccinations expressing your opinions or attitudes; make it start with *: 
                '''
        }


def categorize_reasons(responses):
    def get_prompt(response):
        prompt = f'''
            Example A:
                Response: I've been watching news lately, I don't think vaccines are effective. People still get COVID after vaccinations, so I won't get one.
                
                Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from {"cost, ineffective, distrust_government, distrust_vaccines, low_priority"}
                
                Reason: ineffective
            
            Example B:
                Response: Because I don't trust vaccines and I don't trust English medicines. And I've been avurveda medicines for a long time. So I think I will be cured naturally.
                
                Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from {"cost, ineffective, distrust_government, distrust_vaccines, low_priority"}
                
                Reason: distrust_vaccines
            
            Example C:
                Response: {response}
                
                Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from ["cost", "ineffective", "distrust_government", "distrust_vaccines, "low_priority"]
                
                Reason: 
            
        '''
    prompts = [get_prompt(response) for response in responses]
    reasons = [query_openai(p) for p in prompts]
    reasons = [parse_reasons(r) for r in reasons]


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def query_openai(prompt):
  while True:
    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except openai.APIError:
      continue
    break
  return response.choices[0].message.content







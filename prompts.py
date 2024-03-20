import openai
import backoff
from utils import parse_reasons, REASONS

ATTITUDE_PROMPT = {
            "role": "user",
            "content": '''Based on the lessons you learned and your previous attitude towards COVID vaccinations, choose your current attitude towards COVID vaccinations from {definitely no, probably no, probably yes, and definitely yes}.
                Choice A: definitely no.
                Choice B: probably no.
                Choice C: probably yes.
                Choice D: definitely yes.
                What's your current attitude towards COVID vaccinations from {definitely no, probably no, probably yes, and definitely yes} ?
                Your answer:
            '''
        }

def profile_prompt(profile_str):
    return [{"role": "user",
             "content":f'''
                - Gender:  female 
                - Age:  50 years old
                - Education:  College graduate 
                - Occupation:  small business owner 
                - Political belief:  moderate democrat 
                - Religion:  Buddhist. 
                Attitude towards COVID vaccination from [definitely yes, probably yes, probably no, definitely no] in two words: 
                    probably yes.

                - Gender:  female
                - Age:  27 years old
                - Education:  college degree in science
                - Occupation:  stay-at-home mom
                - Political belief:  Republican
                - Religion:  Baptist
                Attitude towards COVID vaccination from [definitely yes, probably yes, probably no, definitely no] in two words: 
                    probably no.

                {profile_str}
                Attitude towards COVID vaccination from in two words:
                '''
             }]

def news_policies_prompt(news, policies = None, top_k=5):
    prompt = {
        "role": "user",
        "content": f"You read following news today about COVID:\n {news}\n "
    }
    if policies != None:
        policy_prompt = f"The government has also issued the following policies:\n {policies}\n"
        prompt['content'] += policy_prompt
    question = f"What have you learned? Summarize {top_k} lessons you have learned: "
    prompt['content'] += question

    return prompt

def tweets_prompt(tweets, k=5):
    TWEETS_PROMPT = {
        "role": "user",
        "content": f"You read following tweets about COVID:\n {tweets}\n What have you learned? Summarize {k} lessons you have learned: "
    }
    return TWEETS_PROMPT

ENDTURN_REFLECTION_PROMPT = {
            "role": "user",
            "content": '''Based on your background and the lessons you have learned, 
                elaborate on why you are hesitant towards getting vaccinations: '''
        }

REFLECTION_PROMPT = {
            "role": "user",
            "content": '''Based on your background and the lessons you have learned, 
                reflect upon the most significant reasons that cause your attitude towards COVID vaccination to change or stay unchanged: '''
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
                
                Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from {REASONS}
                
                Reason: ineffective
            
            Example B:
                Response: Because I don't trust vaccines and I don't trust English medicines. And I've been avurveda medicines for a long time. So I think I will be cured naturally.
                
                Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from {REASONS}
                
                Reason: distrust_vaccines
            
            
            Response: {response}
            
            Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from {REASONS}
            
            Reason: 
        '''
        return prompt
    
    prompts = [get_prompt(response) for response in responses]
    print(prompts)
    reasons = [query_openai(p) for p in prompts]
    # for p in prompts:
    #     if p != "":
    #         reasons.append("")
    #     else:
    #         reasons.append(query_openai(p))
    reasons = [parse_reasons(r) for r in reasons]
    return reasons


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai(prompt):
  while True:
    try:
      response = openai.chat.completions.create(
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

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai_messages(messages, model="gpt-3.5-turbo"):
  while True:
    try:
      response = openai.chat.completions.create(
        model=model,
        messages=messages,
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







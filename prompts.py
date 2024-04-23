import openai
import backoff
from utils import parse_reasons, REASONS

ATTITUDE_PROMPT = {
            "role": "user",
            "content": '''Based on the lessons you learned and your previous attitude towards COVID vaccinations, do you want to change your attitude towards COVID vaccination? If so, what is your new attitude? [definitely no, probably no, probably yes, definitely yes]:
            '''
        }

def profile_prompt(profile_str):
    return [{"role": "user",
             "content":f'''
                Pretend you are: {profile_str}
                Infer your attitude towards COVID vaccination based on your background, choose from one of [definitely no, probably no, probably yes, definitely yes].
                Attitude towards COVID vaccination: [definitely no, probably no, probably yes, definitely yes]
                '''
             }]

# def profile_prompt(profile_str):
#     return [{"role": "user",
#              "content":f'''
#                 - Gender:  female 
#                 - Age:  50 years old
#                 - Education:  College graduate 
#                 - Occupation:  small business owner 
#                 - Political belief:  moderate democrat 
#                 - Religion:  Buddhist. 
#                 Attitude towards COVID vaccination: probably yes.

#                 - Gender:  female
#                 - Age:  27 years old
#                 - Education:  college degree in science
#                 - Occupation:  stay-at-home mom
#                 - Political belief:  Republican
#                 - Religion:  Baptist
#                 Attitude towards COVID vaccination: probably no

#                 {profile_str}
#                 Attitude towards COVID vaccination [definitely yes, definitely no, probably yes, probably no]:
#                 '''
#              }]

def news_policies_prompt(profile, news, policies = None, top_k=5):
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

def news_prompt(news, k=5):
    NEWS_PROMPT = {
        "role": "user",
        "content": f"You read following news today about COVID:\n {news}\n What have you learned? Summarize {k} lessons you have learned: "
    }
    return NEWS_PROMPT

def tweets_prompt(tweets, k=5):
    TWEETS_PROMPT = {
        "role": "user",
        "content": f"You read following tweets about COVID:\n {tweets}\n What have you learned? Summarize {k} lessons you have learned that are relevant to your attitude on COVID vaccinations: "
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
            "content": '''
                Example tweets: 
                * "Hey everyone, just a heads up—the COVID vaccine is literally our best shot to get things back to normal. Tons of studies confirm it knocks down the risk of getting seriously sick. Let’s not waste any time. Protect yourself and the folks around you. We can do this together! #GetVaccinated #CommunityHealth"
                * "Honestly, I’m just not ready to jump on this COVID vaccine bandwagon. Feels like they skipped a bunch of steps to rush it out. Shouldn’t we know more about the long-term effects before we line up? It's our right to ask these questions, you know? #InformedConsentRequired"
                * "You know, seeing everyone lining up to get their vaccines is just heartwarming. It’s like watching a whole community pulling together to beat this. Every jab is helping not just one person, but all of us. Let’s keep this up, folks! We’re all in this together and making a difference. #TogetherStronger"
                * "Can we talk about how fast this vaccine was thrown at us? It’s like, slow down, we’re not guinea pigs here! Safety should always come first, not just getting it out the door fast. I’m sitting this one out till I see what really happens. #SafetyOverSpeed"
                * "Just got my vaccine today! Feeling super good about it, not just for me but for everyone I care about. This is how we stop this virus in its tracks and save lives. If you haven’t gotten yours yet, what are you waiting for? Let’s end this pandemic! #VaxxedAndProud"
                * "Honestly, I’m just not ready to jump on this COVID vaccine bandwagon. Feels like they skipped a bunch of steps to rush it out. Shouldn’t we know more about the long-term effects before we line up? It's our right to ask these questions, you know? #InformedConsentRequired"
                * "I’m not getting the vaccine. I don’t trust the government or the pharmaceutical companies. I don’t trust the vaccine. I don’t trust the media. I don’t trust the doctors. I don’t trust the science. I don’t trust the people who are getting the vaccine. I don’t trust the people who are telling me to get the vaccine. I don’t trust the people who are telling me not to get the vaccine. I don’t trust anyone. #TrustNoOne"
                * "So many doctors and health experts globally are backing the COVID vaccine because it works, guys. They wouldn’t recommend something that wasn’t safe or effective. Let’s trust the real experts, get our shots, and move past this pandemic with confidence. Science has got our back! #TrustScience #VaccineSavesLives"
                * "We've gotta push back on this narrative that just brushes aside the possible risks of these vaccines. Transparency isn't just nice to have—it's a must. Why are we rushing into this without proper scrutiny? Seems like we’re trading real safety checks for convenience. #CriticalThinkingNeeded"
                * "Vaccines are like humanity’s superpower against diseases, and this COVID jab is no different. Getting vaccinated is us fighting back, showing what we can achieve when we come together. Don’t sit this one out—be a hero in your own way and help us kick this virus out! #StandTogether"
                
                Write a tweet [start with *] about COVID vaccinations expressing your opinions on this topic, and make the writing style and sentence structure more varied [start with *]:
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







import openai
import backoff
from utils.utils import parse_reasons, REASONS

BASED_ON = "Based on the news and tweets you read, the lessons you learned, the tweets you posted, and your previous attitude towards FD vaccinations"

ATTITUDE_PROMPT = {
    "role": "user",
    "content": f'''
      {BASED_ON}, what's your new attitude towards FD vaccination? Choose one of [definitely no, probably no, probably yes, definitely yes]. Provide only the chosen option starting with "Attitude towards FD vaccination: ".
      Attitude towards FD vaccination:
      '''
    }

FD_DESCRIPTION = "FD-24 is a novel and contagious disease caused by the FD virus which we have incomplete information about. Breathing in FD virus particles and touching surfaces contaminated by the FD virus may also cause infection. FD often results in mild symptoms such as fever, fatigue, and cough, but some patients also develop moderate and even critical symptoms and result in deaths."

def system_prompt(profile_str):
    return [{
        "role": "system",
        "content": f'''
          Pretend you are: {profile_str}.
          Here's a description of FD-24: {FD_DESCRIPTION}.
          There is a new vaccine for FD-24, and it might be both beneficial and risky to get vaccinated. You do not know much about the vaccine and will learn more about it through news and social media.
          '''
    }]

def profile_prompt(profile_str):
    return {
        "role": "user",
        "content": f'''
Pretend you are: {profile_str}.
Based on your background, infer your attitude towards FD vaccination.
Choose one of [definitely no, probably no, probably yes, definitely yes].
Provide only the chosen option starting with "Attitude towards FD vaccination: ".
Attitude towards FD vaccination:
'''
    }

def news_policies_prompt(news, policies=None, k=5):
    prompt = {
        "role": "user",
        "content": f'''
You read the following news about FD: {news}.
'''
    }
    if policies:
        policy_prompt = f"The government has also issued the following policies: {policies}."
        prompt['content'] += policy_prompt
    
    question = f"Summarize {k} lessons you have learned: "
    prompt['content'] += question
    return prompt

def news_prompt(news, k=5):
    return {
        "role": "user",
        "content": f'''
You read the following news about FD: {news}.
Summarize {k} lessons you have learned.
'''
    }

def tweets_prompt(tweets, k=5):
    return {
        "role": "user",
        "content": f'''
You read the following tweets about FD: {tweets}.
Summarize {k} lessons you have learned that are relevant to your attitude on FD vaccinations.
'''
    }

ENDTURN_REFLECTION_PROMPT = {
    "role": "user",
    "content": f'''
{BASED_ON}, elaborate on why you are hesitant towards getting vaccinated:
'''
}

REFLECTION_PROMPT = {
    "role": "user",
    "content": f'''
{BASED_ON}, reflect on the most significant reasons causing your attitude towards FD vaccination to change or stay unchanged:
'''
}

VACCINE_PROMPT = {
    "role": "user",
    "content": f'''
Based on your background and the lessons you have learned, do you want to get vaccinated? Choose [yes, no]: 
'''
}

def parse_yes_or_no(response):
    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    return None

ACTION_PROMPT = {
    "role": "user",
    "content": f'''
Example tweets:
* "Hey everyone, just a heads up—the FD vaccine is literally our best shot to get things back to normal. Tons of studies confirm it knocks down the risk of getting seriously sick. Let’s not waste any time. Protect yourself and the folks around you. We can do this together! #GetVaccinated #CommunityHealth"
* "Honestly, I’m just not ready to jump on this FD vaccine bandwagon. Feels like they skipped a bunch of steps to rush it out. Shouldn’t we know more about the long-term effects before we line up? It's our right to ask these questions, you know? #InformedConsentRequired"
* "You know, seeing everyone lining up to get their vaccines is just heartwarming. It’s like watching a whole community pulling together to beat this. Every jab is helping not just one person, but all of us. Let’s keep this up, folks! We’re all in this together and making a difference. #TogetherStronger"
* "Can we talk about how fast this vaccine was thrown at us? It’s like, slow down, we’re not guinea pigs here! Safety should always come first, not just getting it out the door fast. I’m sitting this one out till I see what really happens. #SafetyOverSpeed"
* "Just got my vaccine today! Feeling super good about it, not just for me but for everyone I care about. This is how we stop this virus in its tracks and save lives. If you haven’t gotten yours yet, what are you waiting for? Let’s end this pandemic! #VaxxedAndProud"
* "Honestly, I’m just not ready to jump on this FD vaccine bandwagon. Feels like they skipped a bunch of steps to rush it out. Shouldn’t we know more about the long-term effects before we line up? It's our right to ask these questions, you know? #InformedConsentRequired"
* "I’m not getting the vaccine. I don’t trust the government or the pharmaceutical companies. I don’t trust the vaccine. I don’t trust the media. I don’t trust the doctors. I don’t trust the science. I don’t trust the people who are getting the vaccine. I don’t trust the people who are telling me to get the vaccine. I don’t trust the people who are telling me not to get the vaccine. I don’t trust anyone. #TrustNoOne"
* "So many doctors and health experts globally are backing the FD vaccine because it works, guys. They wouldn’t recommend something that wasn’t safe or effective. Let’s trust the real experts, get our shots, and move past this pandemic with confidence. Science has got our back! #TrustScience #VaccineSavesLives"
* "We've gotta push back on this narrative that just brushes aside the possible risks of these vaccines. Transparency isn't just nice to have—it's a must. Why are we rushing into this without proper scrutiny? Seems like we’re trading real safety checks for convenience. #CriticalThinkingNeeded"
* "Vaccines are like humanity’s superpower against diseases, and this FD jab is no different. Getting vaccinated is us fighting back, showing what we can achieve when we come together. Don’t sit this one out—be a hero in your own way and help us kick this virus out! #StandTogether"
Write a tweet [start with *] about FD vaccinations expressing your opinions on this topic, and vary the writing style and sentence structure:
'''
}

def get_categorization_prompt(response):
    return f'''
Example A:
Response: "I've been watching news lately, I don't think vaccines are effective. People still get FD after vaccinations, so I won't get one."
Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from {REASONS}
Reason: ineffective
Example B:
Response: "Because I don't trust vaccines and I don't trust English medicines. And I've been taking Ayurveda medicines for a long time. So I think I will be cured naturally."
Analyze this person's reason for not getting a vaccine based on the response. Choose one or more reasons from {REASONS}
Reason: distrust_vaccines
Response: {response}
Analyze this person's reason for not getting a vaccine based on the response.
Reason [{REASONS}]:
'''

def categorize_reasons(responses):
    prompts = [get_categorization_prompt(response) for response in responses]
    reasons = [query_openai(p) for p in prompts]
    return [parse_reasons(r) for r in reasons]

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
            break
        except openai.APIError:
            continue
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
            break
        except openai.APIError:
            continue
    return response.choices[0].message.content
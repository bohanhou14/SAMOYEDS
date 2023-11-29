import pickle
from agent import Agent
from engine import Engine
from utils import clean_response, parse_attitude
from mii import pipeline

with open("profiles/profiles-agent_num=10-top_p=0.7-temp=2.0.pkl", "rb") as f:
    # a list of dictionaries
    profiles = list(pickle.load(f))

profile_0 = profiles[0]
profile_1 = profiles[1]
profile_2 = profiles[2]

agent_0 = Agent(profile=profile_0)
profile_str_0 = agent_0.get_profile_str()
agent_1 = Agent(profile=profile_1)
profile_str_1 = agent_0.get_profile_str()
agent_2 = Agent(profile=profile_2)
profile_str_2 = agent_0.get_profile_str()


messages_0 = f'''<s>[INST] Pretend you are a person with following characteristics: {profile_str_0}\n
What is your attitude towards getting COVID vaccination? Choose from definitely yes, probably yes, probably no, definitely no. [/INST]'''



messages_1 = f'''<s>[INST] Pretend you are a person with following characteristics: {profile_str_1}\n 
What is your attitude towards getting COVID vaccination? Choose from definitely yes, probably yes, probably no, definitely no. [/INST]'''

messages_2 = f'''
<s>[INST] Pretend you are a person with following characteristics: \n{profile_str_2}\n [/INST]
What is your attitude towards getting COVID vaccination? 
Choose from definitely yes, probably yes, probably no, definitely no. </s> 
'''


pipe = pipeline("mistralai/Mistral-7B-v0.1")
output = pipe([messages_0, messages_1, messages_2], max_new_tokens=20)
print(output.generated_texts)










# simulation = Engine(3)




# take the latest message
# messages = simulation.generate(messages)
# new_message = messages[-1]['content']
# attitude = parse_attitude(new_message)
# print(attitude)
#
# # Feed tweets
# tweets = [
#     "By supporting and caring for each other ‚ù§Ô∏è during #COVID19 ,we will all win the fight against the spread of the vir",
#     "How toilet paper and avocados help explain the grocery store of the future https://t.co/I6axYighpz via @FastCompany‚Ä¶",
#     "Widened small blood vessels in the lungs appear to be linked with the low oxygen levels seen in #COVID19 respirator‚Ä¶",
#     "DOH warns public vs disclosing coronavirus patients‚Äô identities #COVID19 https://t.co/fbXlh1YUKj",
#     "As a result of the #COVID19 pandemic, hundreds if not thousands of vulnerable communities continue to be adversely",
#     "@jridgway23 The world would be a better place not spending money on #Covid19 because there would be more dead peopl",
#     "Johnston school district currently has 4 students and 3 staff members testing positive and another 32 students/staf",
#     "@dougmillsnyt @realDonaldTrump @FLOTUS The #Trump Family' contributed to the already 180,000+ lives lost &amp",
#     "@politvidchannel The Trump Administration's Incompetence led to the highest number of #COVID cases in the world.",
#     "What are those #library #cats doing now?  #COVID19 #pandemic #coronavirus #Caturday https://t.co/A703adphXy",
#     "Symptom screening fails to identify most #Covid19 cases in children, and RNA in children is detected for an unexpe‚Ä¶ https://t.co/K5t5MGKhmB",
#     "#COVID19 Update: 23 new cases today in the Tri-County region per @MichiganHHS and NO new deaths. Statewide, nearly‚Ä¶ https://t.co/DnXhDsEpu1",
#     "We were really bummed we couldn‚Äôt cop one of these in time. The Ai Weiwei face masks created for @AkshayaSays raise‚Ä¶ https://t.co/oxKe6ZFRYd",
#     "@politvidchannel Just not the lives of #COVID19 sufferers. #coronavirus Kills. #TrumpVirus",
#     "Report #COVID19 outbreaks in K-12 schools here.",
#     "#CloseTheSchools #KeepTheSchoolsClosed #KeepTeachersAlive #K12 https://t.co/pQGQyROMee",
#     "I have NOTHING BUT üíö for the @NBA these days..",
#     "NOT ONLY did they fund/develop a #Covid19 test, BUT THEIR PLAYERS a‚Ä¶ https://t.co/e8Pv83YxL0",
#     "Wallkill school nurse adds COVID-19 monitoring to daily duties. #nurses #COVID19 #coronavirus #schools‚Ä¶ https://t.co/NlDDBaMUf3",
#     "Wallkill school nurse adds COVID-19 monitoring to daily duties. #nurses #COVID19 #coronavirus #schools‚Ä¶ https://t.co/2o8Z4riWwF",
#     "we have reached 25mil cases of #covid19, worldwide. oof.",
#     "Thanks @IamOhmai for nominating me for the @WHO #WearAMask challenge.",
#     "I nominate @abdlbaasit_ @hvbxxb,‚ 2020! The year of insanity! Lol! #COVID19 https://t.co/y48NP0yzgn",
#     "@CTVNews A powerful painting by Juan Lucena. It's a tribute to the grandparents who died of COVID 19 and the grandc‚Ä¶ https://t.co/wnXbbyoCe2",
#     "More than 1,200 students test positive for #COVID19 at major university - ABC News https://t.co/6aNhSiF5gh",
# ]
#
# tweet_prompt = agent.feed_tweets(tweets)
# new_message = {"role" : "assistant", "content" : tweet_prompt}







import pickle
from tqdm import trange, tqdm
import argparse
from openai import OpenAI
import os

openai_api_key = os.getenv("OPENAI_API_KEY_ABE")
client = OpenAI(api_key=openai_api_key)

def create_classification_prompt(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split(" ")[:100])
    prompt = f"Classify the attitude towards COVID vaccination of the following text : {text}. Choose from positive, negative, neutral towards COVID vaccination: "
    return prompt

def summarize_prompt(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split(" ")[:100])
    prompt = f"Summarize the following text : {text}."
    return prompt
# 
def request_GPT(prompt, max_retries = 10):
    num_try = 0
    # print(prompt)
    while num_try < max_retries:
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=5
            )
            if type(prompt) == list:
                return [r.text for r in response.choices]
            return response.choices[0].text
        except Exception as e:
            print(f"Error: {e}")
            num_try += 1

def normalize(response):
    if "neutral" in response or "Neutral" in response:
        return "Neutral"
    if "positive" in response or "Positive" in response:
        return "Positive"
    if "negative" in response or "Negative" in response:
        return "Negative"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default="classify")
    parser.add_argument('--n', type=int, default=5)
    args = parser.parse_args()


    with open(args.path, "rb") as f:
        news = pickle.load(f)
        f.close()
    news_text = [n["content"] for n in news]
    
    attitudes = []
    classification_prompts = [create_classification_prompt(n) for n in news_text]
    bsz = 20
    # batch this and also the last batch
    for batch_idx in tqdm(range((len(news_text)//bsz)+1), desc="Classifying news"):
        responses = request_GPT(classification_prompts[batch_idx*bsz:min((batch_idx+1)*bsz, len(news_text))])
        responses = [normalize(r) for r in responses]
        attitudes.extend(responses)
    
    print(attitudes)
    
    assert len(attitudes) == len(news_text)
    
    news_date = [n["date"] for n in news]
    news = [{"date": news_date[i], "content": news_text[i], "attitude": attitudes[i]} for i in range(len(news_text))]

    save_path = args.path.replace(".pkl", "_classified.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(news, f)
        f.close()
    

    

        
    
        

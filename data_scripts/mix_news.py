import pickle
import numpy as np
from sandbox.news import News
np.random.seed(42)

if __name__ == "__main__":

    neg_life = "/home/bhou4/SAMOYEDS/data/news-neg_life-k=100.pkl"
    pos_life = "/home/bhou4/SAMOYEDS/data/news-pos_life-k=100.pkl" 
    neg_vac = "/home/bhou4/SAMOYEDS/data/news-neg_vac-k=100.pkl"
    pos_vac = "/home/bhou4/SAMOYEDS/data/news-pos_vac-k=100.pkl"

    with open(neg_life, "rb") as f:
        neg_life_data = pickle.load(f)
        f.close()
    with open(pos_life, "rb") as f:
        pos_life_data = pickle.load(f)
        f.close()
    with open(neg_vac, "rb") as f:
        neg_vac_data = pickle.load(f)
        f.close()
    with open(pos_vac, "rb") as f:
        pos_vac_data = pickle.load(f)
        f.close()
    pos_news = pos_life_data + pos_vac_data
    neg_news = neg_life_data + neg_vac_data

    pos_news_data = [News(text, "positive") for text in pos_news]
    neg_news_data = [News(text, "negative") for text in neg_news]
    news_data = pos_news_data + neg_news_data
    
    np.random.shuffle(pos_news)
    np.random.shuffle(neg_news)
    np.random.shuffle(news_data)


    with open(f"/home/bhou4/SAMOYEDS/data/news-mixed-k={len(news_data)}.pkl", "wb") as f:
        pickle.dump(news_data, f)
        f.close()
    with open(f"/home/bhou4/SAMOYEDS/data/news-pos-k={len(pos_news)}.pkl", "wb") as f:
        pickle.dump(pos_news, f)
        f.close()
    with open(f"/home/bhou4/SAMOYEDS/data/news-neg-k={len(neg_news)}.pkl", "wb") as f:
        pickle.dump(neg_news, f)
        f.close()
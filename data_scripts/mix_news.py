import pickle
import numpy as np
np.random.seed(42)

if __name__ == "__main__":
    neg_life = "data/news-neg_life-k=100.pkl"
    pos_life = "data/news-pos_life-k=100.pkl" 
    neg_vac = "data/news-neg_vac-k=100.pkl"
    pos_vac = "data/news-pos_vac-k=100.pkl"

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
    news = pos_news + neg_news
    np.random.shuffle(pos_news)
    np.random.shuffle(neg_news)
    np.random.shuffle(news)
    with open(f"data/news-k={len(news)}.pkl", "wb") as f:
        pickle.dump(news, f)
        f.close()
    with open(f"data/news-pos-k={len(pos_news)}.pkl", "wb") as f:
        pickle.dump(pos_news, f)
        f.close()
    with open(f"data/news-neg-k={len(neg_news)}.pkl", "wb") as f:
        pickle.dump(neg_news, f)
        f.close()

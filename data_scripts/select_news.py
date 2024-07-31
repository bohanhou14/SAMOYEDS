import pickle

if __name__ == '__main__':
    with open('data/realnews_classified.pkl', 'rb') as f:
        realdata = pickle.load(f)
        f.close()

    with open('data/fakenews_classified.pkl', 'rb') as f:
        fakedata = pickle.load(f)
        f.close()
    
    neu_news = []
    neu_dates = []
    pos_news = []
    pos_dates = []
    neg_news = []
    neg_dates = []
    for i in range(len(realdata)):
        if realdata[i]["attitude"] == "Neutral":
            neu_news.append(realdata[i]["content"])
            neu_dates.append(realdata[i]["date"])
        elif realdata[i]["attitude"] == "Positive":
            pos_news.append(realdata[i]["content"])
            pos_dates.append(realdata[i]["date"])
        else: # Negative
            neg_news.append(realdata[i]["content"])
            neg_dates.append(realdata[i]["date"])
    with open('data/realnews_neutral.pkl', 'wb') as f:
        pickle.dump([{"date": neu_dates[i], "content": neu_news[i]} for i in range(len(neu_news))], f)
        f.close()
    with open('data/realnews_positive.pkl', 'wb') as f:
        pickle.dump([{"date": pos_dates[i], "content": pos_news[i]} for i in range(len(pos_news))], f)
        f.close()
    with open('data/realnews_negative.pkl', 'wb') as f:
        pickle.dump([{"date": neg_dates[i], "content": neg_news[i]} for i in range(len(neg_news))], f)
        f.close()

    
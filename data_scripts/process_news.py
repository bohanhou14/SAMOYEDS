import pandas as pd
import argparse
from tqdm import trange
import pickle

# refactor code
def load_news_and_dates(news_file):
    df = pd.read_csv(news_file)
    news = []
    dates = []
    # Process the data
    for i in range(len(df)):
        if 'publish_date' in df.columns and df['publish_date'][i] != None and str(df['publish_date'][i]) != 'nan':
            if df['title'][i] != None and str(df['title'][i]) != 'nan' and df['content'][i] != None and str(df['content'][i]) != 'nan':
                news.append(df['title'][i] + " " + df['content'][i])
            elif df['title'][i] != None and str(df['title'][i]) != 'nan':
                news.append(df['title'][i])
            elif df['content'][i] != None and str(df['content'][i]) != 'nan':
                news.append(df['content'][i])
            else:
                continue
            dates.append(df['publish_date'][i])
    print(f"Loaded {len(news)} news articles from {news_file}")
    print(f"Loaded {len(dates)} dates from {news_file}")
    return news, dates


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--real', type=str, default='news.csv')
    # parser.add_argument('--fake', type=str, default='news_processed.csv')
    # args = parser.parse_args()

    # Load the data
    realnews_files = ['archive/NewsRealCOVID-19_5.csv', 'archive/NewsRealCOVID-19_7.csv', 'archive/NewsRealCOVID-19.csv']
    fakenews_files = ['archive/NewsFakeCOVID-19_5.csv', 'archive/NewsFakeCOVID-19_7.csv', 'archive/NewsFakeCOVID-19.csv']
    
    
    realnews = []
    realdates = []
    for f in realnews_files:
        news, dates = load_news_and_dates(f)
        realnews.extend(news)
        realdates.extend(dates)

    fakenews = []
    fakedates = []
    # Process the data
    for f in fakenews_files:
        news, dates = load_news_and_dates(f)
        fakenews.extend(news)
        fakedates.extend(dates)
    
    # Save the data in pkl format with date and content
    realdata = []
    for i in range(len(realnews)):
        realdata.append({"date": realdates[i], "content": realnews[i]})
    with open('data/realnews.pkl', 'wb') as f:
        pickle.dump(realdata, f)

    fakedata = []
    for i in range(len(fakenews)):
        fakedata.append({"date": fakedates[i], "content": fakenews[i]})
    with open('data/fakenews.pkl', 'wb') as f:
        pickle.dump(fakedata, f)
    

        
    
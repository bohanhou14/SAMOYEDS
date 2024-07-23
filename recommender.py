import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import compile_enumerate
from tweet import Tweet

class Recommender:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        self.decay_rate = 0.995
        self.alpha = 0.1 # edge weight

    def encode_items(self, items, is_tweet=False):
        if is_tweet:
            res = [self.model.encode(tweet.text) for tweet in items]
            return res
        else:
            return [self.model.encode(item) for item in items]

    def calculate_scores(self, profile_embedding, tweet_embeddings, tweet_times, tweet_authors, current_day, following=None):
        current_time = current_day
        tweet_ages = np.array([current_time - tweet_time for tweet_time in tweet_times])
        decay_factors = np.power(self.decay_rate, tweet_ages)
        similarities = cosine_similarity([profile_embedding], tweet_embeddings)[0]
        weighted_scores = similarities * decay_factors
        # calculate the scores from the social network
        following_keys = [f[0] for f in following]
        following_values = [f[1] for f in following]
        following_map = {k: v for k, v in zip(following_keys, following_values)}
        edge_scores = np.array([following_map.get(author, 0) for author in tweet_authors]) * self.alpha 
        weighted_scores += edge_scores
        return weighted_scores

    def recommend(self, tweet_objects, current_day, agents, num_recommendations=10):
        if len(tweet_objects) == 0:
            raise ValueError("No tweets to recommend")
        tweet_embeddings = self.encode_items(tweet_objects, is_tweet=(type(tweet_objects[0]) == Tweet))
        tweet_times = [tweet.time for tweet in tweet_objects]  # Extracting time from each tweet
        tweet_authors = [tweet.author for tweet in tweet_objects]
        all_recommendations = []

        for agent in agents:
            agent_tweets_texts = set([tweet.text for tweet in agent.tweets])

            profile_embedding = self.model.encode(agent.get_profile_str())
            scores = self.calculate_scores(profile_embedding, tweet_embeddings, tweet_times, tweet_authors, current_day, agent.following)

            sorted_indices = scores.argsort()[::-1]
            top_tweet_indices = []
            for index in sorted_indices:
                if len(top_tweet_indices) < num_recommendations:
                    if tweet_objects[index].text not in agent_tweets_texts:
                        top_tweet_indices.append(index)

            top_tweets_texts = [tweet_objects[i].text for i in top_tweet_indices]
            top_tweets_texts = compile_enumerate(top_tweets_texts)
            all_recommendations.append(top_tweets_texts)

        return all_recommendations
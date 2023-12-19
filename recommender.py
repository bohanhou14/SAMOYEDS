import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import compile_enumerate


class Recommender:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        self.decay_rate = 0.995

    def encode_items(self, items, is_tweet=False):
        if is_tweet:
            return [self.model.encode(tweet.text) for tweet in items]
        else:
            return [self.model.encode(item) for item in items]

    def calculate_scores(self, profile_embedding, tweet_embeddings, tweet_times, current_day):
        current_time = current_day
        tweet_ages = np.array([current_time - tweet_time for tweet_time in tweet_times])
        decay_factors = np.power(self.decay_rate, tweet_ages)

        similarities = cosine_similarity([profile_embedding], tweet_embeddings)[0]
        weighted_scores = similarities * decay_factors
        return weighted_scores

    def recommend(self, tweet_objects, current_day, agents, num_recommendations=10):
        tweet_embeddings = self.encode_items(tweet_objects, is_tweet=True)
        tweet_times = [tweet.time for tweet in tweet_objects]  # Extracting time from each tweet
        all_recommendations = []

        for agent in agents:
            agent_tweets_texts = set([tweet.text for tweet in agent.tweets])

            profile_embedding = self.model.encode(agent.get_profile_str())
            scores = self.calculate_scores(profile_embedding, tweet_embeddings, tweet_times, current_day)

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
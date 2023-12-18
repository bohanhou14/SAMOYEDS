import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class TweetRecommender:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_items(self, items):
        return [self.model.encode(item) for item in items]

    def calculate_scores(self, profile_embedding, tweet_embeddings):
        similarities = cosine_similarity([profile_embedding], tweet_embeddings)[0]
        return similarities

    def recommend(self, tweets, profiles, num_recommendations=10):
        tweet_embeddings = self.encode_items(tweets)
        all_recommendations = []

        for profile in profiles:
            profile_embedding = self.model.encode(profile)
            scores = self.calculate_scores(profile_embedding, tweet_embeddings)

            top_tweet_indices = scores.argsort()[::-1][:num_recommendations]
            top_tweets = [tweets[i] for i in top_tweet_indices]

            all_recommendations.append(top_tweets)

        return all_recommendations


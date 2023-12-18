import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class TweetRecommender:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_items(self, items):
        return [self.model.encode(item.text if hasattr(item, 'text') else item) for item in items]

    def calculate_scores(self, profile_embedding, tweet_embeddings):
        similarities = cosine_similarity([profile_embedding], tweet_embeddings)[0]
        return similarities

    def recommend(self, tweet_objects, profiles, num_recommendations=10):
        # Encoding only the text attribute of tweets
        tweet_embeddings = self.encode_items(tweet_objects)
        all_recommendations = []

        for profile in profiles:
            profile_embedding = self.model.encode(profile)
            scores = self.calculate_scores(profile_embedding, tweet_embeddings)

            sorted_indices = scores.argsort()[::-1][:num_recommendations]
            top_tweet_objects = [tweet_objects[i] for i in sorted_indices]

            # Extracting text from tweet objects for the recommendation list
            top_tweets_texts = [tweet.text for tweet in top_tweet_objects]
            all_recommendations.append(top_tweets_texts)

        return all_recommendations


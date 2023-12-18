import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Recommender:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_items(self, items, is_tweet=False):
        if is_tweet:
            return [self.model.encode(tweet.text) for tweet in items]
        else:
            return [self.model.encode(item) for item in items]

    def calculate_scores(self, profile_embedding, tweet_embeddings):
        similarities = cosine_similarity([profile_embedding], tweet_embeddings)[0]
        return similarities

    def recommend(self, tweet_objects, agents, num_recommendations=10):
        tweet_embeddings = self.encode_items(tweet_objects, is_tweet=True)
        all_recommendations = []

        for agent in agents:
            agent_tweets_texts = [tweet.text for tweet in agent.tweets]  # Agent's own tweets
            profile_embedding = self.model.encode(agent.get_profile_str())
            scores = self.calculate_scores(profile_embedding, tweet_embeddings)

            sorted_indices = scores.argsort()[::-1]
            top_tweet_indices = []
            for index in sorted_indices:
                if len(top_tweet_indices) < num_recommendations:
                    if tweet_objects[index].text not in agent.tweets:
                        top_tweet_indices.append(index)
            
            top_tweets_texts = [tweet_objects[i].text for i in top_tweet_indices]
            all_recommendations.append(top_tweets_texts)

        return all_recommendations

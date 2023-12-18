import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Recommender:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_items(self, items):
        return [self.model.encode(item) for item in items]

    def calculate_scores(self, profile_embedding, tweet_embeddings):
        similarities = cosine_similarity([profile_embedding], tweet_embeddings)[0]
        return similarities

    def recommend(self, tweets, agents, num_recommendations=10):
        tweet_embeddings = self.encode_items(tweets)
        all_recommendations = []

        for agent in agents:
            agent_tweets = agent.tweets  # Accessing agent's own tweets
            profile_embedding = self.model.encode(agent.get_profile_str())
            scores = self.calculate_scores(profile_embedding, tweet_embeddings)

            sorted_indices = scores.argsort()[::-1]
            top_tweets = []
            for i in sorted_indices:
                if len(top_tweets) < num_recommendations:
                    if tweets[i] not in agent_tweets:
                        top_tweets.append(tweets[i])
                else:
                    break

            all_recommendations.append(top_tweets)

        return all_recommendations

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class Recommender:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', time_decay_rate=0.9):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.indices = None
        self.agents = None 
        self.num_agents = -1
        self.similarity_matrix_3d = None
        self.time_decay_rate = time_decay_rate  # Decay rate is set in the constructor

    # def build_profile_index(self):
    #     # add profile embedding
    #     profile_texts = [a.get_profile_str() for a in self.agents]
    #     profile_embeddings = self.encode_items(profile_texts)
    #     self.profile_indices = [[] for _ in range(len(self.agents))]
    #     for i in range(len(self.indices)):
    #         self.profile_indices[i].append(profile_embeddings[i])

    def build_or_update_tweets_index(self):
        if not self.indices:
            self.indices = [[] for _ in range(len(self.agents))]  # Build an index for each agent 
        # Index dimension should be (num_agents, num_tweets, embedding_dim)
        recent_tweets = [a.get_most_recent_tweets() for a in self.agents]  # List of tweets
        # If there are no tweets
        # self.build_profile_index()
        # self.indices = self.profile_indices
        assert recent_tweets[0] != None, ValueError("No tweets found. Probably the agent has not tweeted yet.")
        tweet_embeddings = self.encode_items(recent_tweets, is_tweet=True)

        # Append new tweet embeddings to each agent's index
        for i in range(len(self.indices)):
            self.indices[i].append(tweet_embeddings[i])

    def encode_items(self, items, is_tweet=False):
        if is_tweet:
            res = self.model.encode([tweet.text for tweet in items])
        else:
            res = self.model.encode(items)
        return res
    
    def build_or_update_similarity_matrix(self):
        pass
    
    def update_recommender(self, agents):
        pass

    def recommend(self, num_recommendations=10):
        # This method needs to be implemented as per your use case
        pass
from recommenders.recommender import Recommender
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NewsRecommender(Recommender):
    def __init__(self, news, model_name = "paraphrase-MiniLM-L6-v2", time_decay_rate=0.95, *args, **kwargs):
        super().__init__(model_name, time_decay_rate=time_decay_rate, *args, **kwargs)
        self.news_text = [n[0] for n in news] # News is a list of news articles, (str, date)
        self.news_date = [n[1] for n in news]
        self.num_news = len(news)
        self.news_indices = None
    
    def build_news_index(self):
        self.news_indices = self.encode_items(self.news_text)
    
    def update_similarity_matrix(self):
        """
        Update the similarity matrix when a new tweet is added to each agent.
        
        :param existing_similarity_matrix: numpy array of shape (k, N-1, N2) or None - the current similarity matrix
        :param tweets_embedding: numpy array of shape (k, N, D) - tweet embeddings for k agents, each with N tweets, each embedding has dimension D.
        :param news_embedding: numpy array of shape (N2, D) - news embeddings for N2 articles.
        :return: updated_similarity_matrix - numpy array of shape (k, N, N2) - updated similarity scores.
        """
        # breakpoint()
        indices = self.profile_indices if self.indices is None or len(self.indices[0]) == 0 else self.indices

        k, N = len(indices), len(indices[0])
        N2 = len(self.news_indices)
        
        # Existing similarity matrix is None if we're starting fresh
        if self.similarity_matrix_3d is None:
            self.similarity_matrix_3d = np.zeros((k, N, N2))
            for i in range(k):
                self.similarity_matrix_3d[i] = cosine_similarity(indices[i], self.news_indices)
        else:
            old_N = self.similarity_matrix_3d.shape[1]
            assert N - 1 == old_N, "Number of tweets should have increased by 1"        

            existing_similarity_matrix = self.similarity_matrix_3d    
            updated_similarity_matrix = np.zeros((k, N, N2))
            updated_similarity_matrix[:, :-1, :] = existing_similarity_matrix
            new_tweet_embeddings = np.array([indices[i][-1] for i in range(k)])
            updated_similarity_matrix[:, -1, :] = cosine_similarity(new_tweet_embeddings, self.news_indices)
            self.similarity_matrix_3d = updated_similarity_matrix
    
    def sample_top_k_news_for_agent(self, agent_index, k):
        """
        Sample the top k news articles for an agent based on the cosine similarity between the agent's tweets and the news articles.
        
        :param agent_index: int - the index of the agent
        :param k: int - the number of news articles to recommend
        :return: list of tuples - the top k news articles for the agent, each tuple contains the index of the news article and the similarity score.
        """
        assert self.similarity_matrix_3d is not None, "Please build or update the similarity matrix first"
        assert agent_index < self.num_agents, "Invalid agent index"
        
        similarities = self.similarity_matrix_3d[agent_index]
        top_k_indices = np.argpartition(similarities, -k, axis=None)[-k:]
        top_k_indices = np.unravel_index(top_k_indices, similarities.shape)
        top_k_similarities = similarities[top_k_indices]
        top_k_news = [(i, s) for i, s in zip(top_k_indices[1], top_k_similarities)]
        return top_k_news
    
    def update_recommender(self, agents):
        self.agents = agents
        self.num_agents = len(agents)
        self.build_or_update_tweets_index()
        if self.news_indices is None:
            self.build_news_index()
        if self.profile_indices is None:
            self.build_profile_index()
        self.update_similarity_matrix()

    def recommend(self, agents, num_recommendations=10):
        self.update_recommender(agents)
        recommendations = []
        for i in range(self.num_agents):
            top_news = self.sample_top_k_news_for_agent(i, num_recommendations)
            recommendations.append(top_news) # (index of news article, similarity score)
        return recommendations

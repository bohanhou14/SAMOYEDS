from recommenders.recommender import Recommender
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NewsRecommender(Recommender):
    def __init__(self, model_name = "paraphrase-MiniLM-L6-v2", time_decay_rate=0.95, *args, **kwargs):
        super().__init__(model_name, time_decay_rate=time_decay_rate, *args, **kwargs)
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

        k, N = len(self.indices), len(self.indices[0])
        N2 = len(self.news_indices)
        
        # Existing similarity matrix is None if we're starting fresh
        if self.similarity_matrix_3d is None:
            self.similarity_matrix_3d = np.zeros((k, N, N2))
            for i in range(k):
                self.similarity_matrix_3d[i] = cosine_similarity(self.indices[i], self.news_indices)
        else:
            old_N = self.similarity_matrix_3d.shape[1]
            # breakpoint()
            assert N - 1 == old_N, f"Number of tweets should have increased by 1, print(similarity_matrix_3d.shape) {self.similarity_matrix_3d.shape}"        

            existing_similarity_matrix = self.similarity_matrix_3d    
            updated_similarity_matrix = np.zeros((k, N, N2))
            updated_similarity_matrix[:, :-1, :] = existing_similarity_matrix
            new_tweet_embeddings = np.array([self.indices[i][-1] for i in range(k)])
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
        
        top_k_news_text = [self.news_text[i] for i in top_k_indices[1]]
        top_k_news_stance = [self.news_stance[i] for i in top_k_indices[1]]
        top_k_news_sim = [s for s in top_k_similarities]
        return top_k_news_text, top_k_news_stance, top_k_news_sim
    
    def update_recommender(self, agents, news_data):
        self.agents = agents
        self.num_agents = len(agents)
        self.build_or_update_tweets_index()
        self.news_text = [n.text for n in news_data] # News is a list of news articles, (str, date)
        self.news_stance = [n.stance for n in news_data]
        self.num_news = len(news_data)
        self.build_news_index()
        self.update_similarity_matrix()
        self.news_indices = None # update to None so next time it recommends new news article

    def recommend(self, news_data, agents, num_recommendations=10):
        if agents[0].get_most_recent_tweets() == None:
            return news_data[:num_recommendations]
        self.update_recommender(agents, news_data)
        recommendations = []
        for i in range(self.num_agents):
            top_news = self.sample_top_k_news_for_agent(i, num_recommendations)
            recommendations.append(top_news) # (news_text, news_stance, similarity score)
        return recommendations

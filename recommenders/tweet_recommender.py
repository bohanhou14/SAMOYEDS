from recommenders.recommender import Recommender
from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_utils import log_info
import numpy as np

class TweetRecommender(Recommender):
    def __init__(self, model_name = 'paraphrase-MiniLM-L6-v2', time_decay_rate=0.95, alpha=0.1, *args, **kwargs):
        super().__init__(model_name, time_decay_rate, *args, **kwargs)
        self.alpha = alpha # weight of following relation

    
    def build_or_update_similarity_matrix(self):
        """
        Compute the similarity matrix for all agents' tweets.
        # similarity_matrix_3d[0][1][2][4] == similarity_matrix_3d[1][0][4][2]
        # or: similarity between Agent 0's 3rd tweet and Agent 1's 5th tweet
        :return: None but updates the similarity_matrix_3d attribute
        """
        assert len(self.indices) > 0, "Please build or update the index first"

        num_tweets = len(self.indices[0])

        if self.similarity_matrix_3d is None:
            self.similarity_matrix_3d = np.zeros((self.num_agents, self.num_agents, num_tweets, num_tweets))
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        sim = cosine_similarity(self.indices[i], self.indices[j])
                        self.similarity_matrix_3d[i, j] = sim 
                        
        else:  # Update similarity matrix
            extended_similarity_matrix_3d = np.pad(self.similarity_matrix_3d, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant', constant_values=0)
            new_embeddings = np.array([self.indices[i][-1] for i in range(len(self.indices))])

            # Apply decay to all existing similarities
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:  # Skip self-similarity if not needed
                        sim_i_to_j = cosine_similarity([new_embeddings[i]], [self.indices[j][-1]])
                        sim_j_to_i = cosine_similarity([new_embeddings[j]], [self.indices[i][-1]])

                        # decay the old similarities
                        extended_similarity_matrix_3d[i, j] *= self.time_decay_rate
                        extended_similarity_matrix_3d[j, i] *= self.time_decay_rate

                        # Update similarity matrix with new similarities and decay factor
                        # breakpoint()
                        extended_similarity_matrix_3d[i, j, -1, :-1] = sim_i_to_j * self.time_decay_rate
                        extended_similarity_matrix_3d[j, i, :-1, -1] = sim_j_to_i.T * self.time_decay_rate
            
            # don't compute self-similarity
            self.similarity_matrix_3d = extended_similarity_matrix_3d
        
        # Apply following relation
        self.extend_following_similarity_graph()
        self.similarity_matrix_3d += self.following_graph

    def sample_top_k_sim_of_an_agent_new_tweets(self, agent_index, k):
        """
        Sample the top k similarities between the new tweets of an agent and the tweets of other agents.
        :param agent_index: int - the index of the agent
        :param k: int - the number of similarities to recommend
        :return: list of tuples - the top k similarities for the agent, each tuple contains the index of the agent, the index of the tweet, and the similarity score.
        """

        relevant_slice = self.similarity_matrix_3d[agent_index, :, -1, :] # get the similarities of new tweets of agent at agent_index with all other agents

        # Flatten the slice to simplify finding the top_k values
        flat_slice = relevant_slice.flatten()
        
        # Determine the top-K indices in the flattened array
        top_k_indices_flat = np.argpartition(flat_slice, -k)[-k:]
        top_k_indices_flat = top_k_indices_flat[np.argsort(-flat_slice[top_k_indices_flat])]
        
        # Get dimensions of the relevant slice
        num_agents = relevant_slice.shape[0]
        num_tweets = relevant_slice.shape[1]
        
        # Convert flat indices back to (agent, tweet) indices
        top_k_indices = np.unravel_index(top_k_indices_flat, (num_agents, num_tweets))
        
        # Zip the indices with the corresponding similarity values
        top_k_values = [(top_k_indices[0][i], top_k_indices[1][i], flat_slice[top_k_indices_flat[i]]) for i in range(k)]
        
        return top_k_values
    
    def extend_following_similarity_graph(self):
        """
        Build a following similairty graph from the agents' following lists.
        It's not to build new following graph, but to extend the existing one with a new shape
        :return: list of lists - the following graph
        """
        self.following_graph = np.zeros(self.similarity_matrix_3d.shape)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    # if agent i follows agent j, add a similarity score
                    if j in self.agents[i].following:
                        follow_score = self.agents[i].following[j]
                        self.following_graph[i, j, :, :] = np.full(self.following_graph[i, j, :, :].shape, self.alpha * follow_score)

    def update_recommender(self, agents):
        self.agents = agents
        self.num_agents = len(agents)
        with log_info():
            self.build_or_update_tweets_index()
        with log_info():
            self.build_or_update_similarity_matrix()

    def recommend(self, agents, num_recommendations=10):
        self.update_recommender(agents)
        recommendations = []
        all_tweets = [a.get_all_tweets() for a in self.agents]
        # breakpoint()
        for i in range(self.num_agents):
            top_k_values = self.sample_top_k_sim_of_an_agent_new_tweets(i, num_recommendations)
            for j in range(num_recommendations):
                agent_index, tweet_index, similarity = top_k_values[j]
                recommendations.append((i, all_tweets[agent_index][tweet_index].text, similarity))
                # breakpoint()
        # breakpoint()
        return recommendations
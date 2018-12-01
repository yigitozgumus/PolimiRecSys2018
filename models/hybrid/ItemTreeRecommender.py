"""
Author: Semsi Yigit Ozgumus
"""
import numpy as np
from sklearn.preprocessing.data import normalize

from base.BaseRecommender import RecommenderSystem
from base.RecommenderUtils import check_matrix, to_okapi

try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Simimlarity, reverting to Python")
    from base.Similarity_old import Similarity_old

from models.Slim_mark2.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython as Slim_mark2
from models.Slim_mark1.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython as Slim_mark1
from base.Similarity.Compute_Similarity import Compute_Similarity
from sklearn.preprocessing import normalize

class ItemTreeRecommender(RecommenderSystem):

    def __init__(self, URM_train, URM_train_tfidf, ICM, sparse_weights=True):
        super(ItemTreeRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, "csr")
        self.URM_train_tfidf = check_matrix(URM_train_tfidf, "csr")
        self.ICM = check_matrix(ICM, "csr")
        self.sparse_weights = sparse_weights
        self.parameters = None
        self.RECOMMENDER_NAME = "ItemTreeRecommender"

    def __str__(self):
        return "Item Tree 3 Level Hybrid Recommender"

    def fit(self, topK=250, shrink=100, alpha=0.7017094, beta=0.51034483, gamma=0.16206897, normalize=False,
            similarity="jaccard", **similarity_args):
        self.k = topK
        self.shrink = shrink
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.normalize = normalize

        print("Item Tree Hybrid Recommender: Model fitting begins")
        # Calculate all the Similarity Matrices One by one
        # URM tfidf --> 50446 x 50446
        self.sim_URM_tfidf = Compute_Similarity(self.URM_train_tfidf.T, shrink=0, topK=200, normalize=normalize,
                                                similarity=similarity, **similarity_args)
        # ICM tfidf --> 20635 x 20635
        self.ICM = to_okapi(self.ICM)
        self.sim_ICM_tfidf = Compute_Similarity(self.ICM.T, shrink=0, topK=25, normalize=normalize,
                                                similarity=similarity, **similarity_args)
        # URM.T tfidf --> 20635 x 20635
        self.sim_URM_T_tfidf = Compute_Similarity(self.URM_train_tfidf, shrink=10, topK=350, normalize=normalize,
                                        similarity=similarity, **similarity_args)
        # Slim --> 20635 x 20635
        self.sim_Slim = Slim_mark1(self.URM_train)

        if self.sparse_weights:
            # URM
            self.W_sparse_URM = self.sim_URM_tfidf.compute_similarity()
            # UCM
            self.W_sparse_ICM = self.sim_ICM_tfidf.compute_similarity()
            # self.W_sparse_UCM = self.sim_UCM_tfidf.fit()
            # ICM
            self.W_sparse_URM_T = self.sim_URM_T_tfidf.compute_similarity()
            # Slim
            # lambda_i = 0.37142857, lambda_j = 0.97857143
            self.W_sparse_Slim = self.sim_Slim.fit()
        # add the parameters for the logging
        self.parameters = "sparse_weights= {}, similarity= {},shrink= {}, neighbourhood={},normalize= {}, alpha= {}, beta={}, gamma={}".format(
            self.sparse_weights, similarity, shrink, topK, normalize,
            alpha, beta, gamma)

        #self.merged_Ws = alpha * self.W_sparse_ICM + (1 - alpha) * self.W_sparse_Slim

    def recommend(self, playlist_id_array, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False, export=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(playlist_id_array):
            playlist_id_array = np.atleast_1d(playlist_id_array)
            single_user = True
        else:
            single_user = False
        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1
        if self.sparse_weights:
            # First Branch
            #scores_merged = self.URM_train[playlist_id].dot(self.merged_Ws).toarray().ravel()
            scores_ICM = self.URM_train[playlist_id_array].dot(self.W_sparse_ICM).toarray()
            scores_Slim = self.URM_train[playlist_id_array].dot(self.W_sparse_Slim).toarray()
            score_first_branch = self.alpha * scores_ICM + (1- self.alpha) * scores_Slim
            # Second Branch
            scores_URM_T = self.URM_train[playlist_id_array].dot(self.W_sparse_URM_T).toarray()
            scores_second_branch = self.beta * score_first_branch + (1 - self.beta) * scores_URM_T
            #scores_second_branch = self.beta * scores_merged + (1 - self.beta) * scores_URM_T
            # Third Branch
            scores_URM = self.W_sparse_URM[playlist_id_array].dot(self.URM_train).toarray()
            scores = self.gamma * scores_URM + (1 - self.gamma) * scores_second_branch

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            user_profile = self.URM_train[playlist_id_array]
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                # print(rated.shape)
                # print(self.W_sparse.shape)
                den = rated.dot(self.W_sparse).toarray()
            else:
                den = rated.dot(self.W)
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den
        for user_index in range(len(playlist_id_array)):

            user_id = playlist_id_array[user_index]
            if remove_seen_flag:
                scores[user_index, :] = self._remove_seen_on_scores(user_id, scores[user_index, :])

        relevant_items_partition = (-scores).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores[
            np.arange(scores.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            if not export:
                return ranking_list
            elif export:
                return str(ranking_list[0]).strip("[,]")

        if not export:
            return ranking_list
        elif export:
            return str(ranking_list).strip("[,]")


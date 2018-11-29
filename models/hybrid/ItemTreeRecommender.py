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

from models.Slim_BPR.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython

class ItemTreeRecommender(RecommenderSystem):

    def __init__(self,URM_train,URM_train_tfidf, ICM, sparse_weights=True, verbose=False, similarity_mode="jaccard",
                 normalize=False, alpha=0.168, beta=0.317, gamma=0.546):
        super(ItemTreeRecommender,self).__init__()
        self.URM_train = check_matrix(URM_train,"csr")
        self.URM_train_tfidf = check_matrix(URM_train_tfidf,"csr")
        self.ICM = check_matrix(ICM,"csr")
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        self.normalize = normalize
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.parameters = None

    def __str__(self):
        return "Item Tree 3 Level Hybrid Recommender"

    def fit(self, k = 250, shrink= 100, alpha=None, beta= None, gamma = None):
        self.k = k
        self.shrink = shrink
        # Check the parameters for the tuning scenerio
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        print("Item Tree Hybrid Recommender: Model fitting begins")
        # Calculate all the Similarity Matrices One by one
        # URM tfidf --> 50446 x 50446
        self.sim_URM_tfidf = Similarity_old(self.URM_train_tfidf.T,
                                            shrink=0,
                                            verbose=self.verbose,
                                            neighbourhood=200,
                                            mode=self.similarity_mode,
                                            normalize=self.normalize)
        # ICM tfidf --> 20635 x 20635
        self.ICM = to_okapi(self.ICM)
        self.sim_ICM_tfidf = Similarity_old(self.ICM.T,
                                            shrink=0,
                                            verbose=self.verbose,
                                            neighbourhood=25,
                                            mode=self.similarity_mode,
                                            normalize=self.normalize)
        # URM.T tfidf --> 20635 x 20635
        self.sim_URM_T_tfidf = Similarity_old(self.URM_train_tfidf,
                                              shrink=10,
                                              verbose=self.verbose,
                                              neighbourhood=350,
                                              mode=self.similarity_mode,
                                              normalize=self.normalize)
        # Slim --> 20635 x 20635
        self.sim_Slim = Slim_BPR_Recommender_Cython(self.URM_train)

        if self.sparse_weights:
            # URM
            self.W_sparse_URM = normalize(self.sim_URM_tfidf.compute_similarity(),axis=1,norm="l2")
            # UCM
            self.W_sparse_ICM = normalize(self.sim_ICM_tfidf.compute_similarity(),axis=1,norm="l2")
            # self.W_sparse_UCM = self.sim_UCM_tfidf.fit()
            # ICM
            self.W_sparse_URM_T = normalize(self.sim_URM_T_tfidf.compute_similarity(),axis=1,norm="l2")
            # Slim
            self.W_sparse_Slim = normalize(self.sim_Slim.fit(
                lambda_i=0.37142857,
                lambda_j = 0.97857143,
                learning_rate = 0.001,
                epochs=50), axis=1, norm="l2")
        # add the parameters for the logging
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2},shrink= {3}, neighbourhood={4},normalize= {5}, alpha= {6}, beta={7}, gamma={8}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize,
            self.alpha, self.beta, self.gamma)
        self.merged_Ws = self.alpha * self.W_sparse_ICM + (1 - self.alpha) * self.W_sparse_Slim

    def recommend(self, playlist_id, exclude_seen= True, n= None, export = False):
        if n is None:
            n = self.URM_train.shape[1] - 1
        if self.sparse_weights:
            # First Branch
            scores_merged = self.URM_train[playlist_id].dot(self.merged_Ws).toarray().ravel()
            scores_ICM = self.URM_train[playlist_id].dot(self.W_sparse_ICM).toarray().ravel()
            scores_Slim = self.URM_train[playlist_id].dot(self.W_sparse_Slim).toarray().ravel()
            #score_first_branch = self.alpha * scores_ICM + (1- self.alpha) * scores_Slim
            # Second Branch
            scores_URM_T = self.URM_train[playlist_id].dot(self.W_sparse_URM_T).toarray().ravel()
            #scores_second_branch = self.beta * score_first_branch + (1 - self.beta) * scores_URM_T
            scores_second_branch = self.beta * scores_merged + (1- self.beta) * scores_URM_T
            # Third Branch
            scores_URM = self.W_sparse_URM[playlist_id].dot(self.URM_train).toarray().ravel()
            scores = self.gamma * scores_URM + (1-self.gamma) * scores_second_branch

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            user_profile = self.URM_train[playlist_id]
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                # print(rated.shape)
                # print(self.W_sparse.shape)
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den
        if exclude_seen:
            scores = self._remove_seen_on_scores(playlist_id, scores)

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(
            -scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")




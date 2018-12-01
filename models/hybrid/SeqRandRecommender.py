
"""
Author: Semsi Yigit Ozgumus
"""
import numpy as np
from sklearn.preprocessing.data import normalize

from models.Slim_mark1.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython
from base.BaseRecommender import RecommenderSystem
from base.RecommenderUtils import check_matrix
try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Simimlarity, reverting to Python")
    from base.Similarity_old import Similarity_old


class SeqRandRecommender(RecommenderSystem):

    def __init__(self,URM_train,URM_train_tfidf,UCM,ICM,sequential_playlists,sparse_weights=True, verbose=False,similarity_mode="tanimoto",
                 normalize=False, alpha=0.168, beta = 0.375, gamma = 0.717):
        super(SeqRandRecommender,self).__init__()
        self.URM_train = check_matrix(URM_train,"csr")
        self.URM_train_tfidf = check_matrix(URM_train_tfidf,"csr")
        self.UCM = check_matrix(UCM,"csr")
        self.ICM = check_matrix(ICM,"csr")
        self.seq_list = sequential_playlists
        self.sparse_weights = sparse_weights
        self.similarity_mode = similarity_mode
        self.verbose = verbose
        self.normalize = normalize

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.parameters = None

    def __str__(self):
        return "Sequential Random 2 Level Hybrid Recommender"

    def partition(self,first,second,partition):
        denominator = second * partition + first
        return first/denominator, 1 - first/denominator

    def fit(self, k=250, shrink=100, alpha= None, beta= None, gamma = None):
        self.k = k
        self.shrink = shrink
        # check the parameters for the tuning scenerio
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        # Calculate all the Similarity Matrices one by one
        print("Sequential Random Recommender: model fitting begins")
        # URM_tfidf --> 50446 x 50446
        self.sim_URM_tfidf = Similarity_old(self.URM_train_tfidf.T,
                                            shrink=shrink,
                                            verbose=self.verbose,
                                            neighbourhood=k* 2,
                                            mode=self.similarity_mode,
                                            normalize=self.normalize)
        # UCM_tfidf --> 50446 x 50446
        self.sim_UCM_tfidf = Similarity_old(self.UCM.T,
                                            shrink=shrink,
                                            verbose=self.verbose,
                                            neighbourhood= k* 2,
                                            mode= self.similarity_mode,
                                            normalize=True)
        #self.sim_UCM_tfidf = Slim_BPR_Recommender_Cython(self.URM_train_tfidf.T)
        # ICM_tfidf --> 20635 x 20635
        self.sim_ICM_tfidf = Similarity_old(self.ICM.T,
                                            shrink=shrink,
                                            verbose=self.verbose,
                                            neighbourhood=k,
                                            mode=self.similarity_mode,
                                            normalize=self.normalize)
        self.sim_Slim = Slim_BPR_Recommender_Cython(self.URM_train)

        if self.sparse_weights:
            # URM
            self.W_sparse_URM = self.sim_URM_tfidf.compute_similarity()
            # UCM
            self.W_sparse_UCM = self.sim_UCM_tfidf.compute_similarity()
            #self.W_sparse_UCM = self.sim_UCM_tfidf.fit()
            # ICM
            self.W_sparse_ICM = self.sim_ICM_tfidf.compute_similarity()
            # Slim
            self.W_sparse_Slim = self.sim_Slim.fit()
        # add the parameters for the logging
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2},shrink= {3}, neighbourhood={4},normalize= {5}, alpha= {6}, beta={7}, gamma={8}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize, self.alpha,self.beta,self.gamma)

        # calculate the portions
        self.item_seq, self.user_seq = self.partition(self.beta,self.alpha,self.gamma)
        self.user_rand, self.item_rand = self.partition(self.alpha, self.beta,1- self.gamma)

        # Calculate the User based Part
        self.sim_user = self.alpha * self.W_sparse_UCM + (1-self.alpha) * self.W_sparse_URM
        # Calculate the Item based Part
        self.sim_item = self.beta * self.W_sparse_Slim + (1-self.beta) * self.W_sparse_ICM

    def recommend(self,playlist_id,exclude_seen=True,n=None, export=False):
        if n is None:
            n = self.URM_train.shape[1] -1

        if self.sparse_weights:
            # Compute the User based Part
            scores_users = self.sim_user[playlist_id].dot(self.URM_train).toarray().ravel()

            scores_items = self.URM_train[playlist_id].dot(self.sim_item).toarray().ravel()

            scores = None
            if playlist_id in self.seq_list:
                scores = self.item_seq * scores_items + self.user_seq * scores_users
            else:
                scores = self.user_rand * scores_users + self.item_rand * scores_items

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

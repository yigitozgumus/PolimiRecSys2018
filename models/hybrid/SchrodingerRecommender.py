"""
Author: Semsi Yigit Ozgumus
"""
import numpy as np
from sklearn.preprocessing.data import normalize

from base.BaseRecommender import RecommenderSystem
from base.RecommenderUtils import check_matrix

try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Simimlarity, reverting to Python")
    from base.Similarity import Similarity

from models.Slim_BPR.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython

class ScrodingerRecommender(RecommenderSystem):

    def __init__(self,URM_train,URM_train_tfidf,UCM, ICM,sequential_playlists, sparse_weights=True, verbose=False, similarity_mode="jaccard",
                 normalize=False, alpha=0.168, beta=0.317, gamma=0.546, omega = 0.666):
        super(ScrodingerRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train,"csr")
        self.URM_train_tfidf = check_matrix(URM_train_tfidf,"csr")
        self.ICM = check_matrix(ICM,"csr")
        self.UCM = check_matrix(UCM,"csr")
        self.seq_list = sequential_playlists
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        self.normalize = normalize
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.parameters = None

    def __str__(self):
        return "Sequential Random 4 Level Hybrid Recommender"

    def fit(self, k = 250, shrink= 100, alpha=None, beta= None, gamma = None, omega= None):
        self.k = k
        self.shrink = shrink
        # Check the parameters for the tuning scenerio
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if omega is not None:
            self.omega = omega
        print("Sequential Random Hybrid Recommender mark 2: Model fitting begins")
        # Calculate all the Similarity Matrices One by one
        # URM tfidf --> 50446 x 50446
        self.sim_URM_tfidf = Similarity(self.URM_train_tfidf.T,
                                       shrink=0,
                                       verbose=self.verbose,
                                       neighbourhood=200,
                                       mode=self.similarity_mode,
                                       normalize=self.normalize)
        # ICM tfidf --> 20635 x 20635
        self.sim_ICM_tfidf = Similarity(self.ICM.T,
                                        shrink=0,
                                        verbose=self.verbose,
                                        neighbourhood=25,
                                        mode=self.similarity_mode,
                                        normalize=self.normalize)
        # URM.T tfidf --> 20635 x 20635
        self.sim_URM_T_tfidf = Similarity(self.URM_train_tfidf,
                                       shrink=10,
                                       verbose=self.verbose,
                                       neighbourhood=350,
                                       mode=self.similarity_mode,
                                       normalize=self.normalize)
        # Slim --> 20635 x 20635
        self.sim_Slim_item = Slim_BPR_Recommender_Cython(self.URM_train)
        self.sim_Slim_user = Slim_BPR_Recommender_Cython(self.URM_train.T)

        if self.sparse_weights:
            # URM
            self.W_sparse_URM = normalize(self.sim_URM_tfidf.compute_similarity(),axis=1,norm="l2")
            # ICM
            self.W_sparse_ICM = normalize(self.sim_ICM_tfidf.compute_similarity(),axis=1,norm="l2")
            # URM_T
            self.W_sparse_URM_T = normalize(self.sim_URM_T_tfidf.compute_similarity(),axis=1,norm="l2")
            # Slim
            self.W_sparse_Slim_item = normalize(self.sim_Slim_item.fit(
                lambda_i=0.37142857,
                lambda_j=0.97857143,
                learning_rate=0.001,
                epochs=30), axis=1, norm="l2")

            # Slim_T
            self.W_sparse_Slim_user = normalize(self.sim_Slim_user.fit(
                lambda_i=1,
                lambda_j=1,
                learning_rate=0.001,
                epochs=30), axis=1, norm="l2")

        # add the parameters for the logging
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2},shrink= {3}, neighbourhood={4},normalize= {5}, alpha= {6}, beta={7}, gamma={8}, omega={9}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize,
            self.alpha, self.beta, self.gamma, self.omega)

    def recommend(self, playlist_id, exclude_seen= True, n= None, export = False):
        if n is None:
            n = self.URM_train.shape[1] - 1
        if self.sparse_weights:
            #Item First Branch
            scores_ICM = self.URM_train[playlist_id].dot(self.W_sparse_ICM).toarray().ravel()
            scores_Slim_user = self.W_sparse_Slim_user[playlist_id].dot(self.URM_train).toarray().ravel()
            scores_Slim_item = self.URM_train[playlist_id].dot(self.W_sparse_Slim_item).toarray().ravel()
            scores_Slim_final = self.gamma * scores_Slim_user + (1- self.gamma) * scores_Slim_item
            score_first_branch = self.alpha * scores_ICM + (1- self.alpha) * scores_Slim_final
            # Item Second Branch
            scores_URM_T = self.URM_train[playlist_id].dot(self.W_sparse_URM_T).toarray().ravel()
            scores_item_final = self.beta * score_first_branch + (1-self.beta * scores_URM_T)
            # User first Branch
            scores_URM = self.W_sparse_URM[playlist_id].dot(self.URM_train).toarray().ravel()

            # Third Branch
            # Omega should be between 0.5 and 1
            scores = None
            if playlist_id in self.seq_list:
                scores = self.omega * scores_item_final + (1-self.omega) * scores_URM
            else:
                scores = self.omega * scores_URM + (1-self.omega) * scores_item_final

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
            scores = self.filter_seen_on_scores(playlist_id, scores)

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(
            -scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")




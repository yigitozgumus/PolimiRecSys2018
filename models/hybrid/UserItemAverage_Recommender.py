# URM_train version is swapped with tfidfed version
import numpy as np


try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Simimlarity, reverting to Python")
    from base.Similarity_old import Similarity_old

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
from sklearn import feature_extraction

class UserItemAvgRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self,
                 URM_train,
                 UCM,
                 ICM,
                 sparse_weights=True,
                 verbose=True,
                 similarity_mode="cosine",
                 normalize=False,
                 alpha=0.18):
        super(UserItemAvgRecommender, self).__init__()
        self.verbose = verbose
        self.URM_train = check_matrix(URM_train, "csr")
        self.UCM = check_matrix(UCM, "csr")
        self.ICM = check_matrix(ICM, "csr")
        self.sparse_weights = sparse_weights
        self.similarity_mode = similarity_mode
        self.parameters = None
        self.normalize = normalize
        self.alpha = alpha

    def __str__(self):
        return "User Item Weighted Average Collaborative Filtering"

    def fit(self, k=250, shrink=100, alpha=None):
        self.k = k
        self.shrink = shrink
        if not alpha is None:
            self.alpha = alpha
        self.URM_train_T = self.URM_train.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(self.URM_train_T)
        URM_tfidf = URM_tfidf_T.T
        self.URM_tfidf_csr = URM_tfidf.tocsr()
        print("UserItemAvgRecommender: Model fitting begins" )

        # UCM creates a waaay unstable model
        # 50446 * 50446 matrix
        self.similarity_ucm = Similarity_old(self.URM_tfidf_csr.T, shrink=shrink,
                                             verbose=self.verbose,
                                             neighbourhood=k* 2,
                                             mode=self.similarity_mode,
                                             normalize=self.normalize)
        # 20635 * 20635 matrix
        self.similarity_icm = Similarity_old(self.ICM.T, shrink=shrink,
                                             verbose=self.verbose,
                                             neighbourhood=k,
                                             mode=self.similarity_mode,
                                             normalize=self.normalize)
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2},shrink= {3}, neighbourhood={4},normalize= {5}, alpha= {6}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize, self.alpha)

        if self.sparse_weights and self.similarity_mode != "tversky":
           self.W_sparse_UCM = self.similarity_ucm.compute_similarity()
           self.W_sparse_ICM = self.similarity_icm.compute_similarity()
        elif not self.sparse_weights:
           # self.W_UCM = self.similarity_ucm.compute_similarity()
            self.W_ICM = self.similarity_icm.compute_similarity()
            self.W_UCM = self.W_UCM.toarray()
            self.W_ICM = self.W_ICM.toarray()

    def recommend(self, playlist_id, exclude_seen=True, n=None, export=False):
        if n is None:
            n = self.URM_train.shape[1] - 1
        # Compute the scores using the dot product
        if self.sparse_weights:
            scores_ucm = self.W_sparse_UCM[playlist_id].dot(self.URM_train).toarray().ravel()
            scores_icm = self.URM_train[playlist_id].dot(self.W_sparse_ICM).toarray().ravel()
            scores = self.alpha * scores_ucm + (1 - self.alpha) * scores_icm
        else:
            scores_ucm = self.URM_train.T.dot(self.W_UCM[playlist_id])
            scores_icm = self.URM_train.T.dot(self.W_ICM[playlist_id])
            scores = self.alpha * scores_ucm + (1 - self.alpha) * scores_icm
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

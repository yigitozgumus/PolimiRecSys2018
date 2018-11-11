# URM train is swapped with tfidfed version
import numpy as np

from base.Similarity_mark2.tversky import tanimoto_similarity, tversky_similarity
from base.Similarity_mark2.s_plus import dice_similarity, s_plus_similarity, p3alpha_similarity

try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from base.Similarity import Similarity

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix, extract_UCM
from sklearn import feature_extraction

class UserKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train, 
                     UCM, 
                     sparse_weights=True,
                     verbose=True, 
                     similarity_mode="cosine",
                     normalize= False):
        super(UserKNNCFRecommender, self).__init__()
        self.verbose = verbose
        self.URM_train = check_matrix(URM_train, 'csr')
        self.UCM = UCM
        self.sparse_weights = sparse_weights
        self.similarity_mode = similarity_mode
        self.parameters = None
        self.normalize = normalize
        # TFIDF that 
        self.URM_train_T = self.URM_train.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(self.URM_train_T)
        URM_tfidf = URM_tfidf_T.T
        self.URM_tfidf_csr = URM_tfidf.tocsr()
        
    def __str__(self):
        representation = "User KNN Collaborative Filtering " 
        return representation

    def fit(self, k=250, shrink=100):
        self.k = k

        self.shrink = shrink
        if self.similarity_mode != "tversky":
            self.similarity = Similarity(
                self.URM_tfidf_csr.T,
                shrink=shrink,
                verbose=self.verbose,
                neighbourhood=k,
                mode=self.similarity_mode,
                normalize=self.normalize)
        else:
            self.W_sparse = tversky_similarity(self.URM_train, k =k)
            self.W_sparse = check_matrix(self.W_sparse, "csr")
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2}, shrink= {3}, neighbourhood={4}, " \
                          "normalize= {5}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize)
        print("UserKNNCFRecommender: model fitting begins")
        if self.sparse_weights and self.similarity_mode != "tversky":
            self.W_sparse = self.similarity.compute_similarity()
        elif not self.sparse_weights:
            self.W = self.similarity.compute_similarity()
            self.W = self.W.toarray()


    def recommend(self, playlist_id, exclude_seen=True, n=None,filterTopPop=False, export=False):
        if n is None:
            n = self.URM_train.shape[1] - 1

        # compute the scores using the dot product
        if self.sparse_weights:
            scores = self.W_sparse[playlist_id].dot(self.URM_train).toarray().ravel()
        else:
            scores = self.URM_train.T.dot(self.W[playlist_id])
        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            user_profile = self.URM_train[playlist_id]
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den
        if exclude_seen:
            scores = self.filter_seen_on_scores(playlist_id, scores)
        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores)

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")

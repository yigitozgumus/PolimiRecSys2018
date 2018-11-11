import numpy as np

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from base.Similarity import Similarity
from sklearn import feature_extraction

from base.Similarity_mark2.tversky import tversky_similarity


class ItemKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train, sparse_weights=True, verbose=False, similarity_mode="cosine", normalize=False):
        super(ItemKNNCFRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr')
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        self.normalize = normalize
        self.URM_train_T = self.URM_train.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(self.URM_train_T)
        URM_tfidf = URM_tfidf_T.T
        self.URM_tfidf_csr = URM_tfidf.tocsr()
    def __str__(self):
        representation = "Item KNN Collaborative Filtering "
        return representation

    def fit(self, k=250, shrink=100):
        self.k = k
        self.shrink = shrink
        if self.similarity_mode != "tversky":
            self.similarity = Similarity(
                self.URM_tfidf_csr,
                shrink=shrink,
                verbose=self.verbose,
                neighbourhood=k,
                mode=self.similarity_mode,
                normalize=self.normalize)
        else:
            self.W_sparse = tversky_similarity(self.URM_train.T,k=k)
            self.W_sparse = check_matrix(self.W_sparse,"csr")
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2}, shrink= {3}, neighbourhood={4}, normalize={5}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize)
        if self.sparse_weights and self.similarity_mode != "tversky":
            self.W_sparse = self.similarity.compute_similarity()
        elif not self.sparse_weights:
            self.W = self.similarity.compute_similarity()
            self.W = self.W.toarray()


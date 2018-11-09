import numpy as np

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
from base.Similarity import Similarity
import base.Similarity_mark2.s_plus as s_plus


class ItemKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train, sparse_weights=True, verbose=False, similarity_mode="cosine", normalize=False):
        super(ItemKNNCFRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr')
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        self.normalize = normalize

    def __str__(self):
        representation = "Item KNN Collaborative Filtering "
        return representation

    def fit(self, k=250, shrink=100):
        self.k = k
        self.shrink = shrink
        # self.similarity = Similarity(
        #     self.URM_train,
        #     shrink=shrink,
        #     verbose=self.verbose,
        #     neighbourhood=k,
        #     mode=self.similarity_mode,
        #     normalize=self.normalize)
        self.W_sparse = s_plus.tversky_similarity(self.URM_train.T, self.URM_train, k=k, shrink=shrink)
        # self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2}, shrink= {3}, neighbourhood={4}, normalize={5}".format(
        #     self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize)
        # if self.sparse_weights:
        #     self.W_sparse = self.similarity.compute_similarity()
        # else:
        #     self.W = self.similarity.compute_similarity()
        #     self.W = self.W.toarray()

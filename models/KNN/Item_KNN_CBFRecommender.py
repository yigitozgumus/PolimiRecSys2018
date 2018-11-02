import numpy as np
import scipy.sparse as sps

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
from base.Similarity import Similarity


class ItemKNNCBFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train,
                 ICM,
                 sparse_weights=True,
                 verbose=False,
                 similarity_mode="cosine",
                 normalize=False,):
        super(ItemKNNCBFRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr')
        self.ICM = ICM
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        self.normalize = normalize
        self.parameters = None

    def __str__(self):
        representation = "Item KNN Content Based Filtering "
        return representation

    def fit(self, k=250, shrink=100):
        self.k = k
        self.shrink = shrink

        self.similarity = Similarity(self.ICM.T,
                                     shrink=shrink,
                                     verbose=self.verbose,
                                     neighbourhood=k,
                                     mode=self.similarity_mode,
                                     normalize=self.normalize
                                     )
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2}, shrink= {3}, neighbourhood={4}, " \
                          "normalize= {5}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k, self.normalize)

        if self.sparse_weights:
            self.W_sparse = self.similarity.compute_similarity()
        else:
            self.W = self.similarity.compute_similarity()
            self.W = self.W.toarray()

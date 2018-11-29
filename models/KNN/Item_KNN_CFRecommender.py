import numpy as np

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from base.Similarity_old import Similarity_old
from base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr')
        self.sparse_weights = sparse_weights
        self.dataset = None
        self.RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __repr__(self):
        representation = "Item KNN Collaborative Filtering "
        return representation

    def fit(self, topK=350, shrink=10, similarity='jaccard', normalize=False, **similarity_args):
        self.topK = topK
        self.shrink = shrink
        similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)
        self.parameters = "sparse_weights= {0}, similarity= {1}, shrink= {2}, neighbourhood={3}, normalize={4}".format(
            self.sparse_weights, similarity, shrink, topK, normalize)
        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()


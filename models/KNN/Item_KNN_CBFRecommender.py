import numpy as np
import scipy.sparse as sps


from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix, to_okapi,to_tfidf
try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from base.Similarity_old import Similarity_old

from base.Similarity.Compute_Similarity import Compute_Similarity

class ItemKNNCBFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train,ICM,sparse_weights=True,):
        super(ItemKNNCBFRecommender, self).__init__()

        self.RECOMMENDER_NAME = "ItemKNNCBFRecommender"
        self.FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

        self.URM_train = check_matrix(URM_train, 'csr')
        self.ICM = ICM.copy()
        self.sparse_weights = sparse_weights
        self.parameters = None

    def __repr__(self):
        representation = "Item KNN Content Based Filtering "
        return representation

    def fit(self, topK=25, shrink=0, similarity='cosine', normalize=True, feature_weighting="BM25",
            **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = to_okapi(self.ICM)

        elif feature_weighting == "TF-IDF":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = to_tfidf(self.ICM)
        self.parameters = "sparse_weights= {0}, similarity= {1}, shrink= {2}, neighbourhood={3}, " \
                          "normalize= {4}".format(
            self.sparse_weights, similarity, self.shrink, self.topK, normalize)

        similarity = Compute_Similarity(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()


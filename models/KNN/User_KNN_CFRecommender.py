import numpy as np

try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from base.Similarity import Similarity

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix


class UserKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train, sparse_weights=True,verbose=True, similarity_mode="cosine"):
        super(UserKNNCFRecommender, self).__init__()
        self.verbose = verbose
        self.URM_train = check_matrix(URM_train, 'csr')
        self.sparse_weights = sparse_weights
        self.similarity_mode = similarity_mode
        self.parameters = None

    def __str__(self):
        representation = "User KNN Collaborative Filtering " 
        return representation

    def fit(self, k=100, shrink=100,normalize= False):
        self.k = k
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = Similarity(
            self.URM_train.T,
            shrink=shrink,
            verbose=self.verbose,
            neighbourhood=k,
            mode=self.similarity_mode,
            normalize= self.normalize)
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2}, shrink= {3}, neighbourhood={4}".format(
            self.sparse_weights, self.verbose, self.similarity_mode, self.shrink, self.k)

        if self.sparse_weights:
            self.W_sparse = self.similarity.compute_similarity()
        else:
            self.W = self.similarity.compute_similarity()
            self.W = self.W.toarray()

   

# URM train is swapped with tfidfed version
import numpy as np
from sklearn.preprocessing.data import normalize


try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from base.Similarity_old import Similarity_old


from base.Similarity.Compute_Similarity import Compute_Similarity
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix


class UserKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train,
                     sparse_weights=True):
        super(UserKNNCFRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr')
        self.RECOMMENDER_NAME = "UserKNNCFRecommender"
        self.sparse_weights = sparse_weights
        self.parameters = None
        self.dataset = None
        self.compute_item_score = self.compute_score_user_based

        
    def __repr__(self):
        representation = "User KNN Collaborative Filtering " 
        return representation

    # after the tuning k=200, shrink = 0
    def fit(self, topK=200, shrink=0, similarity="jaccard",normalize=False, **similarity_args):
        self.topK = topK
        self.shrink = shrink

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        self.parameters = "sparse_weights= {0}, similarity= {1}, shrink= {2}, neighbourhood={3}, " \
                          "normalize= {4}".format(self.sparse_weights, similarity, shrink, topK, normalize)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()



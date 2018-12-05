
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
from base.RecommenderUtils import check_matrix, to_okapi, to_tfidf
from utils.OfflineDataLoader import OfflineDataLoader

class UserKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    RECOMMENDER_NAME = "UserKNNCFRecommender"
    def __init__(self, URM_train,
                     sparse_weights=True):
        super(UserKNNCFRecommender, self).__init__()
        self.FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
        self.URM_train = check_matrix(URM_train, 'csr')
        self.sparse_weights = sparse_weights
        self.parameters = None
        self.dataset = None
        self.compute_item_score = self.compute_score_user_based

        
    def __repr__(self):
        representation = "User KNN Collaborative Filtering " 
        return representation

    # after the tuning k=200, shrink = 0
    def fit(self, topK=200, shrink=0, similarity="jaccard",normalize=False, feature_weighting="BM25", save_model=False,best_parameters=False, **similarity_args):


        if best_parameters:
            m = OfflineDataLoader()
            folder_path_ucf, file_name_ucf = m.get_parameter(self.RECOMMENDER_NAME)
            self.loadModel(folder_path=folder_path_ucf,file_name=file_name_ucf)
            if feature_weighting == "none":
                similarity = Compute_Similarity(self.URM_train.T, **similarity_args)
            else:
                if feature_weighting == "BM25":
                    self.URM_train_copy = self.URM_train.astype(np.float32)
                    self.URM_train_copy = to_okapi(self.URM_train)

                elif feature_weighting == "TF-IDF":
                    self.URM_train_copy = self.URM_train.astype(np.float32)
                    self.URM_train_copy = to_tfidf(self.URM_train)
                similarity_args = {'asymmetric_alpha': 0.010828193413721543, 'normalize': True, 'shrink': 1000, 'similarity': 'asymmetric', 'topK': 300}
                similarity = Compute_Similarity(self.URM_train_copy.T, **similarity_args)
        else:
            self.topK = topK
            self.shrink = shrink
            self.feature_weighting = feature_weighting
            if self.feature_weighting == "BM25":
                self.URM_train_copy = self.URM_train.astype(np.float32)
                self.URM_train_copy = to_okapi(self.URM_train)

            elif self.feature_weighting == "TF-IDF":
                self.URM_train_copy = self.URM_train.astype(np.float32)
                self.URM_train_copy = to_tfidf(self.URM_train)

            if self.feature_weighting == "none":
                similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                                similarity=similarity, **similarity_args)
            else:
                similarity = Compute_Similarity(self.URM_train_copy.T, shrink=shrink, topK=topK, normalize=normalize,
                                                similarity=similarity, **similarity_args)


        self.parameters = "sparse_weights= {0}, similarity= {1}, shrink= {2}, neighbourhood={3}, " \
                          "normalize= {4}".format(self.sparse_weights, similarity, shrink, topK, normalize)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()
        if save_model:
            self.saveModel("saved_models/submission/",file_name="UserKNNCFRecommender_submission_model")


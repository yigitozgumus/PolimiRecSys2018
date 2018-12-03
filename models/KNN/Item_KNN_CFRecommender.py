import numpy as np

from utils.OfflineDataLoader import OfflineDataLoader
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix, to_okapi, to_tfidf

try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from base.Similarity_old import Similarity_old
from base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCFRecommender(RecommenderSystem, RecommenderSystem_SM):

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()
        self.FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]
        self.URM_train = check_matrix(URM_train, 'csr')
        self.sparse_weights = sparse_weights
        self.dataset = None

    def __repr__(self):
        representation = "Item KNN Collaborative Filtering "
        return representation

    def fit(self, topK=350, shrink=10, similarity='jaccard',feature_weighting="none", normalize=False,save_model=False,best_parameters=False, **similarity_args):

        if best_parameters:
            m = OfflineDataLoader()
            folder_path_icf, file_name_icf = m.get_parameter(self.RECOMMENDER_NAME)
            self.loadModel(folder_path=folder_path_icf,file_name=file_name_icf)
            if self.feature_weighting == "none":
                similarity = Compute_Similarity(self.URM_train, **similarity_args)
            else:
                if self.feature_weighting == "BM25":
                    self.URM_train_copy = self.URM_train.astype(np.float32)
                    self.URM_train_copy = to_okapi(self.URM_train)

                elif self.feature_weighting == "TF-IDF":
                    self.URM_train_copy = self.URM_train.astype(np.float32)
                    self.URM_train_copy = to_tfidf(self.URM_train)
                similarity = Compute_Similarity(self.URM_train_copy, **similarity_args)
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
                similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize,
                                                similarity=similarity, **similarity_args)
            else:
                similarity = Compute_Similarity(self.URM_train_copy, shrink=shrink, topK=topK, normalize=normalize,
                                                similarity=similarity, **similarity_args)
        self.parameters = "sparse_weights= {0}, similarity= {1}, shrink= {2}, neighbourhood={3}, normalize={4}".format(
            self.sparse_weights, similarity, shrink, topK, normalize)
        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()
        if save_model:
            self.saveModel("saved_models/submission/",file_name="ItemKNNCFRecommender_submission_model")


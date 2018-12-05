import numpy as np
import scipy.sparse as sps

from utils.OfflineDataLoader import OfflineDataLoader
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

    RECOMMENDER_NAME = "ItemKNNCBFRecommender"
    def __init__(self, URM_train,ICM,sparse_weights=True,):
        super(ItemKNNCBFRecommender, self).__init__()

        self.FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

        self.URM_train = check_matrix(URM_train, 'csr')
        self.ICM = ICM.copy()
        self.sparse_weights = sparse_weights
        self.parameters = None
        self.dataset = None

    def __repr__(self):
        representation = "Item KNN Content Based Filtering "
        return representation

    def fit(self, topK=600, shrink=1000, similarity='asymmetric', normalize=True, feature_weighting="BM25",save_model=False,best_parameters = False,
            **similarity_args):
        similarity_args = {'asymmetric_alpha': 0.40273209903969387}
        if best_parameters:
            m = OfflineDataLoader()
            folder_path_icbf, file_name_icbf = m.get_parameter(self.RECOMMENDER_NAME)
            self.loadModel(folder_path=folder_path_icbf,file_name=file_name_icbf)
            if feature_weighting == "BM25":
                self.ICM = self.ICM.astype(np.float32)
                self.ICM = to_okapi(self.ICM)

            elif feature_weighting == "TF-IDF":
                self.ICM = self.ICM.astype(np.float32)
                self.ICM = to_tfidf(self.ICM)
            similarity = Compute_Similarity(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize,
                                            similarity=similarity, **similarity_args)
        else:
            self.topK = topK
            self.shrink = shrink
            similarity = Compute_Similarity(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize,
                                            similarity=similarity, **similarity_args)

        self.parameters = "sparse_weights= {0}, similarity= {1}, shrink= {2}, neighbourhood={3}, " \
                          "normalize= {4}".format(
            self.sparse_weights, similarity, self.shrink, self.topK, normalize)



        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

        if save_model:
            self.saveModel("saved_models/submission/",file_name="ItemKNNCBFRecommender_submission_model")


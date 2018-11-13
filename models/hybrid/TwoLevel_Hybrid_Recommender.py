import numpy as np
from sklearn import feature_extraction

from models.Slim_BPR.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
try:
    from base.Cython.Similarity import Similarity
except ImportError:
    print("Unable to load Cython Cosine_Simimlarity, reverting to Python")
    from base.Similarity import Similarity
from base.Similarity_mark2.s_plus import dot_product

class TwoLevelHybridRecommender(RecommenderSystem, RecommenderSystem_SM):
    def __init__(self,URM_train,UCM,ICM,sparse_weights,verbose,similarity_mode,normalize,alpha,avg):
        super(TwoLevelHybridRecommender, self).__init__()
        self.URM_train = URM_train
        self.UCM = UCM
        self.ICM = ICM
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        self.normalize = normalize
        self.alpha = alpha
        self.avg = avg

    def __str__(self):
        return "2 Level Hybrid Recommender"

    def fit(self,k=250, shrink=100, alpha= None,avg= None):
        self.k = k
        self.shrink = shrink
        if not alpha is None:
            self.alpha = alpha
        if not avg is None:
            self.avg = avg

        # Compute the TFIDF of the URM_train
        self.URM_train_T = self.URM_train.T
        URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(self.URM_train_T)
        URM_tfidf = URM_tfidf_T.T
        self.URM_tfidf_csr = URM_tfidf.tocsr()

        # Compute the similarity of the UCM
        self.similarity_ucm = Similarity(self.UCM.T,
                                         shrink = shrink,
                                         verbose = self.verbose,
                                         neighbourhood= k*2,
                                         mode=self.similarity_mode,
                                         normalize=self.normalize)

        # Compute the similarity of ICM
        self.similarity_icm = Similarity(self.ICM.T,
                                         shrink=shrink,
                                         verbose=self.verbose,
                                         neighbourhood=k,
                                         mode=self.similarity_mode,
                                         normalize=self.normalize)
        # Compute the similarity for the Slim
        self.similarity_slim = Slim_BPR_Recommender_Cython(self.URM_train)

        # Add the parameters for the log
        self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2}," \
                          "shrink= {3}, neighbourhood= {4}, normalize= {5}," \
                          "alpha= {6}, average={7}".format(
            self.sparse_weights,
            self.verbose,
            self.similarity_mode,
            self.shrink,
            self.k,
            self.normalize,
            self.alpha,
            self.avg)

        if self.sparse_weights:
            print("2 Level Hybrid Recommender: Similarity computation of UCM is started")
            self.W_sparse_UCM = self.similarity_ucm.compute_similarity()
            print("2 Level Hybrid Recommender: Similarity computation of ICM is started")
            self.W_sparse_ICM = self.similarity_icm.compute_similarity()
            print("2 Level Hybrid Recommender: Similarity computation Slim is started")
            self.W_sparse_slim = self.similarity_slim.fit()
        else:
            print("2 Level Hybrid Recommender: Similarity computation of UCM is started")
            self.W_UCM = self.similarity_ucm.compute_similarity()
            print("2 Level Hybrid Recommender: Similarity computation of ICM is started")
            self.W_ICM = self.similarity_icm.compute_similarity()
            print("2 Level Hybrid Recommender: Similarity computation Slim is started")
            self.W_slim = self.similarity_slim.fit()
            self.W_UCM = self.W_UCM.toarray()
            self.W_ICM = self.W_ICM.toarray()
            self.W_slim = self.W_slim.toarray()
        print(self.W_sparse_UCM.shape, "UCM")
        print(self.W_sparse_ICM.shape, "ICM")
        print(self.W_sparse_slim.shape, "SLIM")


    def recommend(self, playlist_id, exclude_seen=True, n=None, export=False):
        if n is None:
            n = self.URM_train.shape[1]-1

        if self.sparse_weights:
            s_avg = (self.avg * self.W_sparse_ICM) + ((1- self.avg) * self.W_sparse_slim)
            scores_avg = self.URM_train[playlist_id].dot(s_avg).toarray().ravel()
            scores_UCM = self.W_sparse_UCM[playlist_id].dot(self.URM_train).toarray().ravel()
            scores = self.alpha + scores_avg + (1 - self.alpha * scores_UCM)
        else:
            s_avg = (self.avg * self.W_ICM )+ ((1 - self.avg) * self.W_UCM)
            scores_avg = self.URM_tfidf_csr.T.dot(s_avg[playlist_id])
            scores_slim = self.URM_tfidf_csr.T.dot(self.W_slim[playlist_id])
            scores = self.alpha * scores_avg + (1-self.alpha * scores_slim)
        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            user_profile = self.URM_train[playlist_id]
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                # print(rated.shape)
                # print(self.W_sparse.shape)
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den
        if exclude_seen:
            scores = self.filter_seen_on_scores(playlist_id, scores)

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(
            -scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")

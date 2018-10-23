import numpy as np

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
from base.Similarity import Similarity


class ItemKNNCFRecommender(RecommenderSystem,RecommenderSystem_SM):
    
    def __init__(self,URM_train,sparse_weights=True,verbose=False,similarity_mode="cosine"):
        super(ItemKNNCFRecommender,self).__init__()
        self.URM_train = check_matrix(URM_train,'csr')
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        

    def __str__(self):
        representation = "Item KNN Collaborative Filtering "
        return representation

    def fit(self,k=50,shrink=100):
        self.k = k
        self.shrink = shrink
        self.similarity = Similarity(self.URM_train,
        shrink=shrink,
        verbose=self.verbose,
        neighbourhood=k,
        mode=self.similarity_mode)
        self.parameters = self.parameters = "sparse_weights= {0}, verbose= {1}, similarity= {2}".format(
            self.sparse_weights, self.verbose, self.similarity_mode)
        if self.sparse_weights:
            self.W_sparse = self.similarity.compute_similarity()
        else:
            self.W = self.similarity.compute_similarity()
            self.W = self.W.toarray()

    def recommend(self, playlist_id, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False,export = False):
        
        if n==None:
            n=self.URM_train.shape[1]-1

        # compute the scores using the dot product
        if self.sparse_weights:
            user_profile = self.URM_train[playlist_id]

            scores = user_profile.dot(self.W_sparse).toarray().ravel()

        else:

            user_profile = self.URM_train.indices[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]

            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)

        # if self.normalize:
        #     # normalization will keep the scores in the same range
        #     # of value of the ratings in dataset
        #     rated = user_profile.copy()
        #     rated.data = np.ones_like(rated.data)
        #     if self.sparse_weights:
        #         den = rated.dot(self.W_sparse).toarray().ravel()
        #     else:
        #         den = rated.dot(self.W).ravel()
        #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #     scores /= den

        if exclude_seen:
            scores = self._filter_seen_on_scores(playlist_id, scores)

        # if filterTopPop:
        #     scores = self._filter_TopPop_on_scores(scores)

        # if filterCustomItems:
        #     scores = self._filterCustomItems_on_scores(scores)

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")

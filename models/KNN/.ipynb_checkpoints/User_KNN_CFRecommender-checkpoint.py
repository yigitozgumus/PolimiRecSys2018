import numpy as np

from base.Similarity import Similarity
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix

class UserKNNCFRecommender(RecommenderSystem,RecommenderSystem_SM):

    def __init__(self,URM_train,sparse_weights= True):
        super(UserKNNCFRecommender,self).__init__()

        self.URM_train = check_matrix(URM_train,'csr')
        self.sparse_weights = sparse_weights



    def fit(self, k=50, shrink= 100, similarity = 'cosine',normalize = True):
        self.k = k
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = Similarity(self.URM_train,
                                        shrink=shrink,
                                        verbose=True,
                                        neighbourhood=k,
                                        mode=similarity)

        if self.sparse_weights:
            self.W_sparse = self.similarity.computeUUSimilarity()
        else:
            self.W = self.similarity.computeUUSimilarity()
            self.W = self.W.toarray()

    def recommend(self, playlist_id, n=None, exclude_seen=True):
    
        if n == None:
            n = self.URM_train.shape[1] - 1

        # compute the scores using the dot product
        if self.sparse_weights:
            scores = self.W_sparse[playlist_id].dot(self.URM_train).toarray().ravel()
           # print(scores)
        else:
            scores = self.URM_train.T.dot(self.W[playlist_id])

        if exclude_seen:
            scores = self._filter_seen_on_scores(playlist_id, scores)


        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return str(ranking).strip("[]")

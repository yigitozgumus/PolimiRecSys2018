import numpy as np
import time

class RecommenderSystem_SM(object):
    def __init__(self):
        super(RecommenderSystem_SM, self).__init__()
        # self.sparse_weights = None

    def recommend(self, playlist_id, exclude_seen=True, n=None, filterTopPop=False, export=False):

        if n is None:
            n = self.URM_train.shape[1] - 1

        # compute the scores using the dot product
        if self.sparse_weights:
            user_profile = self.URM_train[playlist_id]
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
            # print(scores)
        else:
            # scores = self.URM_train.T.dot(self.W[playlist_id])
            user_profile = self.URM_train.indices[
                self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
            user_ratings = self.URM_train.data[
                self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den

        if exclude_seen:
            scores = self.filter_seen_on_scores(playlist_id, scores)
        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores)

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(
            -scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")


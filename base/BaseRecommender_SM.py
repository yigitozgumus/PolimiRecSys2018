import numpy as np
import time

class RecommenderSystem_SM(object):
    def __init__(self):
        super(RecommenderSystem_SM, self).__init__()
        self.sparse_weights = True
        self.compute_item_score = self.compute_score_item_based

    def compute_score_item_based(self, playlist_id):
        if self.sparse_weights:
            user_profile = self.URM_train.tocsr()[playlist_id]
            return user_profile.dot(self.W_sparse).toarray()
        else:
            user_profile = self.URM_train.indices[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
            relevant_weights = self.W[user_profile]
            return relevant_weights.T.dot(user_ratings)

    def compute_score_user_based(self, user_id):

        if self.sparse_weights:
            return self.W_sparse[user_id].dot(self.URM_train).toarray()
        else:
            # Numpy dot does not recognize sparse matrices, so we must
            # invoke the dot function on the sparse one
            return self.URM_train.T.dot(self.W[user_id])


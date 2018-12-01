import numpy as np
import time
import scipy.sparse as sps
import pickle

from base.RecommenderUtils import check_matrix


class RecommenderSystem_SM(object):
    def __init__(self):
        super(RecommenderSystem_SM, self).__init__()
        self.sparse_weights = True
        self.compute_item_score = self.compute_score_item_based

    def compute_score_item_based(self, playlist_id):
        if self.sparse_weights:
            self.URM_train = check_matrix(self.URM_train,"csr")
            user_profile = self.URM_train[playlist_id]
            return user_profile.dot(self.W_sparse).toarray()
        else:
            result= []
            for playlist in playlist_id:
                user_profile = self.URM_train.indices[self.URM_train.indptr[playlist]:self.URM_train.indptr[playlist + 1]]
                user_ratings = self.URM_train.data[self.URM_train.indptr[playlist]:self.URM_train.indptr[playlist + 1]]
                relevant_weights = self.W[user_profile]
                result.append( relevant_weights.T.dot(user_ratings))
            return np.array(result)

    def compute_score_user_based(self, user_id):

        if self.sparse_weights:
            return self.W_sparse[user_id].dot(self.URM_train).toarray()
        else:
            # Numpy dot does not recognize sparse matrices, so we must
            # invoke the dot function on the sparse one
            return self.URM_train.T.dot(self.W[user_id])

    def saveModel(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))
        dictionary_to_save = {"sparse_weights": self.sparse_weights}
        if self.sparse_weights:
            dictionary_to_save["W_sparse"] = self.W_sparse
        else:
            dictionary_to_save["W"] = self.W
        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))

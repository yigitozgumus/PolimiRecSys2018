import time
from base.evaluation.Metrics import precision, recall, map
from base.RecommenderUtils import check_matrix, removeTopPop
import numpy as np
import pickle

class RecommenderSystem(object):

    def __init__(self):
        super(RecommenderSystem, self).__init__()
        self.RECOMMENDER_NAME ="Abstract Recommender Class"
        self.URM_train = None
        self.URM_test = None
        self.map = None
        self.precision = None
        self.recall = None
        self.parameters = None
        self.sparse_weights = True

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

    def fit(self):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):
        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch

    def _remove_CustomItems_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch

    def _remove_seen_on_scores(self, playlist_id, scores):
        self.URM_train = check_matrix(self.URM_train, "csr")
        seen = self.URM_train.indices[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
        scores[seen] = -np.inf
        return scores

    def get_user_relevant_items(self, playlist_id):
        return self.URM_test.indices[self.URM_test.indptr[playlist_id]:self.URM_test.indptr[playlist_id + 1]]
        # return self.URM_train[playlist_id].indices

    def compute_item_score(self, user_id):
        raise NotImplementedError(
            "Recommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False, export=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_a
        scores_batch = self.compute_item_score(user_id_array)
        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]
            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_CustomItems_flag:
            scores_batch = self._remove_CustomItems_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[ np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[ np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            if not export:
                return ranking_list
            elif export:
                return str(ranking_list[0]).strip("[]")

        if not export:
            return ranking_list
        elif export:
            return str(ranking_list).strip("[]")


    def saveModel(self, folder_path, file_name=None):
        raise NotImplementedError("Recommender: saveModel not implemented")

    def loadModel(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Loading model from file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = pickle.load(open(folder_path + file_name, "rb"))

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        print("{}: Loading complete".format(self.RECOMMENDER_NAME))

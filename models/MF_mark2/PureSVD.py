#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""
import numpy as np
from base.BaseRecommender import RecommenderSystem

from base.RecommenderUtils import check_matrix

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sps


class PureSVDRecommender(RecommenderSystem):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVD"
    def __init__(self, URM_train):
        super(PureSVDRecommender, self).__init__()
        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')
        self.compute_item_score = self.compute_score_SVD
        self.parameters = None

    def __str__(self):
        return "Pure SVD Recommender"

    def fit(self, num_factors=100):
        from sklearn.utils.extmath import randomized_svd
        print("Pure SVD Recommender" + " Computing SVD decomposition...")
        self.U, self.Sigma, self.VT = randomized_svd(self.URM_train,
                                                     n_components=num_factors,
                                                     n_iter=10,
                                                     random_state=None)

        self.s_Vt = sps.diags(self.Sigma) * self.VT
        print(self.s_Vt.shape)
        print("Pure SVD Recommender" + " Computing SVD decomposition... Done!")
        self.parameters = "Number of Factors = {}".format(num_factors)

    def compute_score_SVD(self, user_id_array):
        try:
            item_weights = self.U[user_id_array, :].dot(self.s_Vt)
        except:
            pass
        return item_weights

    def recommend(self, playlist_id_array, remove_seen_flag=True, cutoff=None, remove_CustomItems_flag=False, remove_top_pop_flag=False, export=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(playlist_id_array):
            playlist_id_array = np.atleast_1d(playlist_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        scores = self.compute_score_SVD(playlist_id_array)

        for user_index in range(len(playlist_id_array)):
            user_id = playlist_id_array[user_index]
            if remove_seen_flag:
                scores[user_index, :] = self._remove_seen_on_scores(user_id, scores[user_index, :])

                # relevant_items_partition is block_size x cutoff
            relevant_items_partition = (-scores).argpartition(cutoff, axis=1)[:, 0:cutoff]

            # Get original value and sort it
            # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
            # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
            relevant_items_partition_original_value = scores[
                np.arange(scores.shape[0])[:, None], relevant_items_partition]
            relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
            ranking = relevant_items_partition[
                np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

            ranking_list = ranking.tolist()

            # Return single list for one user, instead of list of lists
            if single_user:
                if not export:
                    return ranking_list
                elif export:
                    return str(ranking_list[0]).strip("[,]")

            if not export:
                return ranking_list
            elif export:
                return str(ranking_list).strip("[,]")

    def saveModel(self, folder_path, file_name=None):

        import pickle
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))
        data_dict = {
            "U": self.U,
            "Sigma": self.Sigma,
            "VT": self.VT,
            "s_Vt": self.s_Vt
        }
        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete")

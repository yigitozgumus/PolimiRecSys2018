#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""
import numpy as np
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sps


class PureSVDRecommender(RecommenderSystem):
    """ PureSVDRecommender"""

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

    def recommend(self, playlist_id, exclude_seen=True, n=None, filterTopPop=False, export=False):
        if n is None:
            n = self.URM_train.shape[1] - 1

        scores = self.compute_score_SVD(playlist_id)

        if exclude_seen:
            scores = self.filter_seen_on_scores(playlist_id, scores)
        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores)

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort( -scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")

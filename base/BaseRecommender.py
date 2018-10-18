"""

@author: Semsi Yigit Ozgumus
"""
import time

from base.Metrics import Metrics
from base.RecommenderUtils import check_matrix
import numpy as np


class RecommenderSystem(object):

    def __init(self):
        """
        Initialization of the class
        """
        super(RecommenderSystem, self).__init__()
        self.URM_train = None
        self.URM_test = None
        self.sparse_weights = None
        self.normalize = None
        # Filter topPop and Custom Items TODO

    def fit(self):
        pass

    def _filter_seen_on_scores(self, playlist_id, scores):
        seen = self.URM_train.indices[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
        scores[seen] = -np.inf
        return scores


    def get_user_relevant_items(self, playlist_id):
        return self.URM_test.indices[self.URM_test.indptr[playlist_id]:self.URM_test.indptr[playlist_id + 1]]

    def recommend(self, playlist_id, n=None, exclude_seen=True, ):
        if n == None:
            n = self.URM_train.shape[1] - 1
        if self.sparse_weights:
            user_profile = self.URM_train[playlist_id]

            scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            user_profile = self.URM_train.indices[
                           self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
            user_ratings = self.URM_train.data[
                           self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]

            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)

        if exclude_seen:
            scores = self._filter_seen_on_scores(playlist_id, scores)
        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return ranking


    def evaluateRecommendations(self,
                                URM_test,
                                at=10,
                                minRatingsPerUser=1,
                                exclude_seen=True,
                                mode="sequential"):  # FilterTopPop Implementation TODO
        URM_test_new = check_matrix(URM_test, format='csr')
        self.URM_test = check_matrix(URM_test_new, format='csr')
        self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen

        numUsers = self.URM_test.shape[0]
        # Prune users with an insufficient number of ratings
        rows = self.URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(numUsers)[mask]
        usersToEvaluate = list(usersToEvaluate)
        if mode == 'sequential':
            return self.evaluateRecommendationsSequential(usersToEvaluate)
        # elif mode == 'parallel':
        # return self.evaluateRecommendationsParallel(usersToEvaluate)
        # elif mode == 'batch':
        # return self.evaluateRecommendationsBatch(usersToEvaluate)
        else:
            raise ValueError("Mode '{}' not available".format(mode))



    def evaluateRecommendationsSequential(self, usersToEvaluate):
        start_time = time.time()
        cumPrecision, cumRecall, cumMap = 0.0, 0.0, 0.0
        num_eval = 0
        metric = Metrics()
        for test_user in usersToEvaluate:
            relevant_items = self.get_user_relevant_items(test_user)
            num_eval += 1
            recommended_items = self.recommend(playlist_id=test_user,
                                               exclude_seen=self.exclude_seen,
                                               n=self.at,
                                               )
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
            cumPrecision += metric.precision(is_relevant)
            cumRecall += metric.recall(is_relevant, relevant_items)
            cumMap += metric.map(is_relevant, relevant_items)

            if num_eval % 100 == 0 or num_eval == len(usersToEvaluate) - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                    num_eval,
                    100.0 * float(num_eval + 1) / len(usersToEvaluate),
                    time.time() - start_time,
                    float(num_eval) / (time.time() - start_time)))

        if num_eval >0 :
            cumPrecision /= num_eval
            cumRecall /= num_eval
            cumMap /= num_eval
            print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
                cumPrecision, cumRecall, cumMap))
        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["precision"] = cumPrecision
        results_run["recall"] = cumRecall
        results_run["map"] = cumMap


        return (results_run)

import time
from tqdm import tqdm
from base.Metrics import precision, recall, map
from base.RecommenderUtils import check_matrix, removeTopPop
import numpy as np
class RecommenderSystem(object):

    def __init__(self):
        super(RecommenderSystem, self).__init__()
        self.URM_train = None
        self.URM_test = None
        self.map = None
        self.precision = None
        self.recall = None
        self.parameters = None
        self.sparse_weights = True
        # Filter topPop and Custom Items TODO

    def fit(self):
        pass

    def filter_seen_on_scores(self, playlist_id, scores):
        self.URM_train = check_matrix(self.URM_train,"csr")
        seen = self.URM_train.indices[self.URM_train.indptr[playlist_id]:self.URM_train.indptr[playlist_id + 1]]
        scores[seen] = -np.inf
        return scores

    def get_user_relevant_items(self, playlist_id):
        return self.URM_test.indices[self.URM_test.indptr[playlist_id]:self.URM_test.indptr[playlist_id + 1]]
        # return self.URM_train[playlist_id].indices

    def _filter_TopPop_on_scores(self, scores):
        scores[self.filterTopPop_ItemsID] = -np.inf
        return scores

    def evaluate_recommendations(self,
                                 URM_test_new,
                                 at=10,
                                 minRatingsPerUser=1,
                                 exclude_seen=True,
                                 mode="sequential",
                                 filterTopPop=False):  # FilterTopPop Implementation TODO
        if filterTopPop != False:
            self.filterTopPop = True

            _, _, self.filterTopPop_ItemsID = removeTopPop(self.URM_train, URM_2=URM_test_new,
                                                           percentageToRemove=filterTopPop)

            print("Filtering {}% TopPop items, count is: {}".format(filterTopPop * 100, len(self.filterTopPop_ItemsID)))

            # Zero-out the items in order to be considered irrelevant
            URM_test_new = check_matrix(URM_test_new, format='lil')
            URM_test_new[:, self.filterTopPop_ItemsID] = 0
            URM_test_new = check_matrix(URM_test_new, format='csr')

            # During testing CSR is faster
        self.URM_test = check_matrix(URM_test_new, format='csr')
        self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen

        self.URM_test = check_matrix(URM_test_new, format='csr')
       # self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen

        numUsers = self.URM_test.shape[0]
        # Prune users with an insufficient number of ratings
        rows = self.URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = minRatingsPerUser <= numRatings
        usersToEvaluate = np.arange(numUsers)[mask]
        usersToEvaluate = list(usersToEvaluate)
        if mode == 'sequential':
            return self.evaluate_recommendations_sequential(usersToEvaluate)
        else:
            raise ValueError("Mode '{}' not available".format(mode))

    def evaluate_recommendations_sequential(self, usersToEvaluate):
        start_time = time.time()
        cum_precision, cum_recall, cum_map = 0.0, 0.0, 0.0
        num_eval = 0
        print("Recommender System: Evaluation for the Test set begins")
        for test_user in usersToEvaluate:
            relevant_items = self.get_user_relevant_items(test_user)
            num_eval += 1
            recommended_items = self.recommend(playlist_id=test_user,
                                               exclude_seen=self.exclude_seen)
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
            cum_precision += precision(is_relevant)
            cum_recall += recall(is_relevant, relevant_items)
            cum_map += map(is_relevant, relevant_items)

            if num_eval % 5000 == 0 or num_eval == len(usersToEvaluate) - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                    num_eval,
                    100.0 * float(num_eval + 1) / len(usersToEvaluate),
                    time.time() - start_time,
                    float(num_eval) / (time.time() - start_time)))

        if num_eval > 0:
            cum_precision /= num_eval
            cum_recall /= num_eval
            cum_map /= num_eval
            self.map = "{:.6f}".format(cum_map)
            self.precision = "{:.6f}".format(cum_precision)
            self.recall = "{:.6f}".format(cum_recall)
            print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
                cum_precision, cum_recall, cum_map))
        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {"precision": cum_precision, "recall": cum_recall, "map": cum_map}

        return (results_run)

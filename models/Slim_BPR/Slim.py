import time, sys

import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet

from base.RecommenderUtils import check_matrix
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.BaseRecommender import RecommenderSystem


class Slim(RecommenderSystem, RecommenderSystem_SM):

    def __init__(self, URM_train, sparse_weights=True, normalize=True):
        super(Slim, self).__init__()
        self.URM_train = URM_train
        self.W_sparse = None
        self.sparse_weights = sparse_weights
        self.normalize = normalize
        self.parameters = None

    def __str__(self):
        return "Slim Recommender with ElasticNet"

    def fit(self, l1_penalty=0.01, l2_penalty=0.01, positive_only=True, topK=100):
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        self.topK = topK
        self.parameters = "sparse_weights= {0},normalize= {1}, l1_penalty= {2}, l2_penalty= {3}, positive_only= {4}".format(
            self.sparse_weights, self.normalize, self.l1_penalty, self.l2_penalty, self.positive_only)

        X = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = X.shape[1]

        # initialize the ElasticNet model
        print("Slim: ElasticNet model fitting begins")
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=True,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)
        print("Slim: Fitting is completed!")
        values, rows, cols = [], [], []
        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):
            # get the target column
            y = X[:, currentItem].toarray()
            # set the j-th column of X to zero
            startptr = X.indptr[currentItem]
            endptr = X.indptr[currentItem + 1]
            bak = X.data[startptr: endptr].copy()
            X.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            # TODO
            self.model.fit(X, y)

            relevant_items_partition = (-self.model.coef_).argpartition(self.topK)[0:self.topK]
            relevant_items_partition_sorting = np.argsort(-self.model.coef_[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            notZerosMask = self.model.coef_[ranking] > 0.0
            ranking = ranking[notZerosMask]

            values.extend(self.model.coef_[ranking])
            rows.extend(ranking)
            cols.extend([currentItem] * len(ranking))

            # finally, replace the original values of the j-th column
            X.data[startptr:endptr] = bak

            if time.time() - start_time_printBatch > 300:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Columns per second: {:.0f}".format(
                    currentItem,
                    100.0 * float(currentItem) / n_items,
                    (time.time() - start_time) / 60,
                    float(currentItem) / (time.time() - start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix(
            (values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)



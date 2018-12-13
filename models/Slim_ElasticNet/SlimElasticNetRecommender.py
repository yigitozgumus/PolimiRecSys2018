#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sps

from utils.OfflineDataLoader import OfflineDataLoader
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import time, sys


class SLIMElasticNetRecommender(RecommenderSystem_SM, RecommenderSystem):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """
    RECOMMENDER_NAME = "Slim_Elastic_Net_Recommender"
    def __init__(self, URM_train):

        RECOMMENDER_NAME = "SLIMElasticNetRecommender"
        super(SLIMElasticNetRecommender, self).__init__()
        self.URM_train = URM_train
        self.parameters = None

    def __repr__(self):
        return "Slim ElasticNet Recommender"

    def fit(self, l1_ratio=0.1, positive_only=True, topK = 400,save_model=False,best_parameters=False, offline=False,submission=False):
        self.parameters = "l1_ratio= {}, topK= {},alpha= {},tol= {},max_iter= {}".format(l1_ratio,topK,0.0001,1e-4,100)
        if offline:
            m = OfflineDataLoader()
            folder, file = m.get_model(self.RECOMMENDER_NAME,training=(not submission))
            self.loadModel(folder_path=folder,file_name=file)
        else:
        
            assert l1_ratio>= 0 and l1_ratio<=1, "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

            self.l1_ratio = l1_ratio
            self.positive_only = positive_only
            self.topK = topK

            # initialize the ElasticNet model
            self.model = ElasticNet(alpha=0.0001,
                                    l1_ratio=self.l1_ratio,
                                    positive=self.positive_only,
                                    fit_intercept=False,
                                    copy_X=False,
                                    precompute=True,
                                    selection='random',
                                    max_iter=100,
                                    tol=1e-4)

           
            URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
            n_items = URM_train.shape[1]
            # Use array as it reduces memory requirements compared to lists
            dataBlock = 10000000
            rows = np.zeros(dataBlock, dtype=np.int32)
            cols = np.zeros(dataBlock, dtype=np.int32)
            values = np.zeros(dataBlock, dtype=np.float32)
            numCells = 0
            start_time = time.time()
            start_time_printBatch = start_time

            # fit each item's factors sequentially (not in parallel)
            for currentItem in tqdm(range(n_items)):
                # get the target column
                y = URM_train[:, currentItem].toarray()
                # set the j-th column of X to zero
                start_pos = URM_train.indptr[currentItem]
                end_pos = URM_train.indptr[currentItem + 1]
                current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
                URM_train.data[start_pos: end_pos] = 0.0
                # fit one ElasticNet model per column
                self.model.fit(URM_train, y)
                nonzero_model_coef_index = self.model.sparse_coef_.indices
                nonzero_model_coef_value = self.model.sparse_coef_.data
                local_topK = min(len(nonzero_model_coef_value)-1, self.topK)
                relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
                relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
                ranking = relevant_items_partition[relevant_items_partition_sorting]

                for index in range(len(ranking)):
                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))
                    rows[numCells] = nonzero_model_coef_index[ranking[index]]
                    cols[numCells] = currentItem
                    values[numCells] = nonzero_model_coef_value[ranking[index]]
                    numCells += 1
                # finally, replace the original values of the j-th column
                URM_train.data[start_pos:end_pos] = current_item_data_backup

                if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                    print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                                      currentItem+1,
                                      100.0* float(currentItem+1)/n_items,
                                      (time.time()-start_time)/60,
                                      float(currentItem)/(time.time()-start_time)))
                    sys.stdout.flush()
                    sys.stderr.flush()
                    start_time_printBatch = time.time()

            # generate the sparse weight matrix
            self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                           shape=(n_items, n_items), dtype=np.float32)
        if save_model:
            self.saveModel("saved_models/submission/",file_name=self.RECOMMENDER_NAME)




import multiprocessing
from multiprocessing import Pool
from functools import partial
import threading


class MultiThreadSLIM_ElasticNet(SLIMElasticNetRecommender, RecommenderSystem_SM):

    def __init__(self, URM_train):
        super(MultiThreadSLIM_ElasticNet, self).__init__(URM_train)

    def __str__(self):
        return "SLIM_mt (l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
        )

    def _partial_fit(self, currentItem, X, topK):
        model = ElasticNet(alpha=1.0,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=False,
                           copy_X=False,
                           precompute=True,
                           selection='random',
                           max_iter=100,
                           tol=1e-4)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        # nnz_idx = model.coef_ > 0.0

        relevant_items_partition = (-model.coef_).argpartition(topK)[0:topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        notZerosMask = model.coef_[ranking] > 0.0
        ranking = ranking[notZerosMask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        #
        # values = model.coef_[nnz_idx]
        # rows = np.arange(X.shape[1])[nnz_idx]
        # cols = np.ones(nnz_idx.sum()) * currentItem
        #
        return values, rows, cols

    def fit(self, l1_penalty=0.1,
            l2_penalty=0.1,
            positive_only=True,
            topK=100,
            workers=multiprocessing.cpu_count()):
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        self.topK = topK

        self.workers = workers

        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = self.URM_train.shape[1]
        # fit item's factors in parallel
        # oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, X=self.URM_train, topK=self.topK)

        # creo un pool con un certo numero di processi
        pool = Pool(processes=self.workers)

        # avvio il pool passando la funzione (con la parte fissa dell'input)
        # e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)
        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        return self.values, self.rows, self.cols

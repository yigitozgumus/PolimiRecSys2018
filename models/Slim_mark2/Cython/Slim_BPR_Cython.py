#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils.OfflineDataLoader import OfflineDataLoader
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import similarityMatrixTopK
from base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


import subprocess
import os, sys, time

import numpy as np
from base.evaluation.Evaluator import SequentialEvaluator

class Slim_BPR_Recommender_Cython(RecommenderSystem_SM, RecommenderSystem, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "SLIM_BPR_Recommender_mark2"
    def __init__(self, URM_train, positive_threshold=1, URM_validation = None,
                 recompile_cython = False, final_model_sparse_weights = True, train_with_sparse_weights = False,
                 symmetric = True):

        super(Slim_BPR_Recommender_Cython, self).__init__()

        self.URM_train = URM_train.copy()
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        self.positive_threshold = positive_threshold

        self.train_with_sparse_weights = train_with_sparse_weights
        self.sparse_weights = final_model_sparse_weights
        self.parameters =None

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        if self.train_with_sparse_weights:
            self.sparse_weights = True
        self.URM_mask = self.URM_train.copy()
        self.URM_mask.data = self.URM_mask.data >= self.positive_threshold
        self.URM_mask.eliminate_zeros()
        assert self.URM_mask.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"
        self.symmetric = symmetric
        if not self.train_with_sparse_weights:
            n_items = URM_train.shape[1]
            requiredGB = 8 * n_items**2 / 1e+06
            if symmetric:
                requiredGB /=2

            print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} items is {:.2f} MB".format(n_items, requiredGB))
        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def __repr__(self):
        return "Slim BPR with Cython mark 2"

    def fit(self, epochs=100, logFile=None,
            batch_size = 1000, lambda_i = 1e-4, lambda_j =1e-4, learning_rate = 0.025, topK = 200,
            sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
            stop_on_validation = False, lower_validatons_allowed = 5, validation_metric="MAP",
            evaluator_object = None, validation_every_n = 1,save_model=False,best_parameters=False, offline=True,submission=False):
        self.parameters = "epochs={0}, batch_size={1}, lambda_i={2}, lambda_j={3}, learning_rate={4}, topK={5}, sgd_mode={6" \
                            "}, gamma={7}, beta_1={8}, beta_2={9},".format(epochs,batch_size,lambda_i,lambda_j,
                                                                        learning_rate,topK,sgd_mode,gamma,beta_1,beta_2)
        if offline:
            m = OfflineDataLoader()
            folder,file = m.get_model(self.RECOMMENDER_NAME,training=(not submission))
            self.loadModel(folder_path=folder,file_name=file)
        else:
            # Import compiled module
            from models.Slim_mark2.Cython.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch
            # Select only positive interactions
            URM_train_positive = self.URM_train.copy()
            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
            URM_train_positive.eliminate_zeros()

            self.sgd_mode = sgd_mode
            self.epochs = epochs
            
            self.cythonEpoch = SLIM_BPR_Cython_Epoch(
                self.URM_mask,
                train_with_sparse_weights = self.train_with_sparse_weights,
                final_model_sparse_weights = self.sparse_weights,
                topK=topK,
                learning_rate=learning_rate,
                li_reg = lambda_i,
                lj_reg = lambda_j,
                batch_size=1,
                symmetric = self.symmetric,
                sgd_mode = sgd_mode,
                gamma=gamma,
                beta_1=beta_1,
                beta_2=beta_2)

            if(topK != False and topK<1):
                raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
            self.topK = topK

            if validation_every_n is not None:
                self.validation_every_n = validation_every_n
            else:
                self.validation_every_n = np.inf

            if evaluator_object is None and stop_on_validation:
                evaluator_object = SequentialEvaluator(self.URM_validation, [10])

            self.batch_size = batch_size
            self.lambda_i = lambda_i
            self.lambda_j = lambda_j
            self.learning_rate = learning_rate


            self._train_with_early_stopping(epochs, validation_every_n, stop_on_validation,
                                        validation_metric, lower_validatons_allowed, evaluator_object,
                                        algorithm_name = self.RECOMMENDER_NAME)


            self.get_S_incremental_and_set_W()

            sys.stdout.flush()
        if save_model:
            self.saveModel("saved_models/submission/",file_name=self.RECOMMENDER_NAME)

    def _initialize_incremental_model(self):
        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()


    def _update_incremental_model(self):
        self.get_S_incremental_and_set_W()


    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self, num_epoch):
       self.cythonEpoch.epochIteration_Cython()


    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
        else:
            if self.sparse_weights:
                self.W_sparse = similarityMatrixTopK(self.S_incremental, k = self.topK)
            else:
                self.W = self.S_incremental


    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "models/SLIM_BPR/Cython"
        #fileToCompile_list = ['Sparse_Matrix_CSR.pyx', 'SLIM_BPR_Cython_Epoch.pyx']
        fileToCompile_list = ['SLIM_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]

            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass

        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))



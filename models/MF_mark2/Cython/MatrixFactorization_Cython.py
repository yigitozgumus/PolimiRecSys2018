from base.BaseRecommender import RecommenderSystem

from base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

import subprocess
import os, sys
import numpy as np


class MatrixFactorization_Cython(RecommenderSystem, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "MatrixFactorization_Cython_Recommender"

    def __init__(self, URM_train, positive_threshold=4, URM_validation=None, recompile_cython=False,
                 algorithm="MF_BPR"):
        super(MatrixFactorization_Cython, self).__init__()
        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        self.algorithm = algorithm
        self.positive_threshold = positive_threshold
        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        self.compute_item_score = self.compute_score_MF
        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def compute_score_MF(self, user_id):

        scores_array = np.dot(self.W[user_id], self.H.T)
        return scores_array

    def recommend(self, playlist_id, n=None, exclude_seen=True, filterTopPop=False, filterCustomItems=False,
                  export=False):
        if n == None:
            n = self.URM_train.shape[1] - 1

        scores_array = self.compute_score_MF(playlist_id)

        if self.normalize:
            raise ValueError("Not implemented")
        if exclude_seen:
            scores = self.filter_seen_on_scores(playlist_id, scores_array)
        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores_array)
        if filterCustomItems:
            scores = self._filterCustomItems_on_scores(scores_array)

        relevant_items_partition = (-scores_array).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        if not export:
            return ranking
        elif export:
            return str(ranking).strip("[]")

    def fit(self, epochs=300, batch_size=1000, num_factors=50,
            learning_rate=0.001, sgd_mode='sgd', user_reg=0.0, positive_reg=0.0, negative_reg=0.0,
            stop_on_validation=False, lower_validatons_allowed=5, validation_metric="MAP",
            evaluator_object=None, validation_every_n=5):
        self.num_factors = num_factors
        self.sgd_mode = sgd_mode
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if evaluator_object is None and stop_on_validation:
            self.evaluate_recommendations(self.URM_validation)

        from models.MF_mark2.Cython.MatrixFactorization_Cython_Epoch import MatrixFactorization_Cython_Epoch

        if self.algorithm == "FUNK_SVD":

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(
                self.URM_train,
                algorithm=self.algorithm,
                n_factors=self.num_factors,
                learning_rate=learning_rate,
                batch_size=1,
                sgd_mode=sgd_mode,
                user_reg=user_reg,
                positive_reg=positive_reg,
                negative_reg=0.0)

        elif self.algorithm == "ASY_SVD":

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(
                self.URM_train,
                algorithm=self.algorithm,
                n_factors=self.num_factors,
                learning_rate=learning_rate,
                batch_size=1,
                sgd_mode=sgd_mode,
                user_reg=user_reg,
                positive_reg=positive_reg,
                negative_reg=0.0)

        elif self.algorithm == "MF_BPR":

            # Select only positive interactions
            URM_train_positive = self.URM_train.copy()

            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
            URM_train_positive.eliminate_zeros()

            assert URM_train_positive.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(
                URM_train_positive,
                algorithm=self.algorithm,
                n_factors=self.num_factors,
                learning_rate=learning_rate,
                batch_size=1,
                sgd_mode=sgd_mode,
                user_reg=user_reg,
                positive_reg=positive_reg,
                negative_reg=negative_reg)

        self._train_with_early_stopping(epochs, validation_every_n, stop_on_validation,
                                        validation_metric, lower_validatons_allowed, evaluator_object,
                                        algorithm_name=self.algorithm)

        self.W = self.W_best
        self.H = self.H_best

        sys.stdout.flush()

    def _initialize_incremental_model(self):

        self.W_incremental = self.cythonEpoch.get_W()
        self.W_best = self.W_incremental.copy()
        self.H_incremental = self.cythonEpoch.get_H()
        self.H_best = self.H_incremental.copy()

    def _update_incremental_model(self):

        self.W_incremental = self.cythonEpoch.get_W()
        self.H_incremental = self.cythonEpoch.get_H()
        self.W = self.W_incremental
        self.H = self.H_incremental

    def _update_best_model(self):
        self.W_best = self.W_incremental.copy()
        self.H_best = self.H_incremental.copy()

    def _run_epoch(self, num_epoch):
        self.cythonEpoch.epochIteration_Cython()

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/models/MF_mark2/Cython"
        fileToCompile_list = ['MatrixFactorization_Cython_Epoch.pyx']

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

                output = subprocess.check_output(' '.join(command), shell=True,
                                                 cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass

        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        # python compileCython.py MatrixFactorization_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a MatrixFactorization_Cython_Epoch.pyx


class MatrixFactorization_BPR_Cython(MatrixFactorization_Cython):
    """
    Subclas allowing only for MF BPR
    """

    RECOMMENDER_NAME = "MatrixFactorization_BPR_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_BPR_Cython, self).__init__(*pos_args, algorithm="MF_BPR", **key_args)

    def fit(self, **key_args):
        super(MatrixFactorization_BPR_Cython, self).fit(**key_args)


class MatrixFactorization_FunkSVD_Cython(MatrixFactorization_Cython):
    """
    Subclas allowing only for FunkSVD
    """

    RECOMMENDER_NAME = "MatrixFactorization_FunkSVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_FunkSVD_Cython, self).__init__(*pos_args, algorithm="FUNK_SVD", **key_args)

    def fit(self, **key_args):
        if "reg" in key_args:
            key_args["positive_reg"] = key_args["reg"]
            del key_args["reg"]

        super(MatrixFactorization_FunkSVD_Cython, self).fit(**key_args)


class MatrixFactorization_AsySVD_Cython(MatrixFactorization_Cython):
    """
    Subclas allowing only for AsySVD
    """

    RECOMMENDER_NAME = "MatrixFactorization_AsySVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_AsySVD_Cython, self).__init__(*pos_args, algorithm="ASY_SVD", **key_args)

    def fit(self, **key_args):
        super(MatrixFactorization_AsySVD_Cython, self).fit(**key_args)

from base.BaseRecommender import RecommenderSystem

from base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from base.RecommenderUtils import check_matrix
import subprocess
import os, sys
import numpy as np
import scipy.sparse as sps
import pickle

class MatrixFactorization_Cython(RecommenderSystem, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "MatrixFactorization_Cython_Recommender"

    def __init__(self, URM_train, positive_threshold=1, URM_validation=None, recompile_cython=False,
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

    def recommend(self, playlist_id_array, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,remove_CustomItems_flag=False, export=False):

         # If is a scalar transform it in a 1-cell array
        if np.isscalar(playlist_id_array):
            playlist_id_array = np.atleast_1d(playlist_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_a
        scores_batch = self.compute_score_MF(playlist_id_array)
        for user_index in range(len(playlist_id_array)):

            user_id = playlist_id_array[user_index]
            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
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

    def fit(self, epochs=300, batch_size=1000, num_factors=100,
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

        #self.W = sps.csr_matrix(self.W_best)
        #self.H = sps.csr_matrix(self.H_best)
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
    def saveModel(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))
        dictionary_to_save = {"W": self.W,
                              "H": self.H}
        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))


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

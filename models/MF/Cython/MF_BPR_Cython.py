#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from base.BaseRecommender import RecommenderSystem
import subprocess
import os, sys
import time
import numpy as np


class MF_BPR_Cython(RecommenderSystem):


    def __init__(self, URM_train, recompile_cython = False):
        super(MF_BPR_Cython, self).__init__()
        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")


    def fit(self, epochs=100,
             URM_test=None,
            filterTopPop = False,
            filterCustomItems = np.array([], dtype=np.int),
            minRatingsPerUser=1,
            batch_size = 1000,
            validate_every_N_epochs = 1,
            start_validation_after_N_epochs = 0,
            num_factors=100,
            positive_threshold=1,
            learning_rate = 0.0025,
            sgd_mode='adagrad',
            user_reg = 0.0,
            positive_reg = 0.005,
            negative_reg = 0.0025):


        self.num_factors = num_factors
        self.positive_threshold = positive_threshold
        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()
        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()
        self.sgd_mode = sgd_mode
        # Import compiled module
        from models.MF.Cython.MF_BPR_Cython_Epoch import MF_BPR_Cython_Epoch
        self.cythonEpoch = MF_BPR_Cython_Epoch(URM_train_positive,
                                                 n_factors = self.num_factors,
                                                 learning_rate=learning_rate,
                                                 batch_size=1,
                                                 sgd_mode = sgd_mode,
                                                 user_reg=user_reg,
                                                 positive_reg=positive_reg,
                                                 negative_reg=negative_reg)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        start_time_train = time.time()
        for currentEpoch in range(epochs):
            start_time_epoch = time.time()
            if self.batch_size>0:
                self.epochIteration()
            else:
                print("No batch not available")
            if (URM_test is not None) and (currentEpoch % validate_every_N_epochs == 0) and \
                            currentEpoch >= start_validation_after_N_epochs:

                print("Evaluation begins")
                self.W = self.cythonEpoch.get_W()
                self.H = self.cythonEpoch.get_H()

                results_run = self.evaluate_recommendations(URM_test)

                self.writeCurrentConfig(currentEpoch, results_run)
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))
            # Fit with no validation
            else:
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))
        # Ensure W and H are up to date
        self.W = self.cythonEpoch.get_W()
        self.H = self.cythonEpoch.get_H()
        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))
        sys.stdout.flush()

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/models/MF/Cython"
        fileToCompile_list = ['MF_BPR_Cython_Epoch.pyx']
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
        # Command to run compilation script
        #python compileCython.py MF_BPR_Cython_Epoch.pyx build_ext --inplace
        # Command to generate html report
        #subprocess.call(["cython", "-a", "MF_BPR_Cython_Epoch.pyx"])

    def epochIteration(self):
        self.cythonEpoch.epochIteration_Cython()

    def writeCurrentConfig(self, currentEpoch, results_run):
        current_config = {'learn_rate': self.learning_rate,
                          'num_factors': self.num_factors,
                          'batch_size': 1,
                          'epoch': currentEpoch}
        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        sys.stdout.flush()


    def recommendBatch(self, users_in_batch, n=None, exclude_seen=True):

        # compute the scores using the dot product
        user_profile_batch = self.URM_train[users_in_batch]
        scores_array = np.dot(self.W[users_in_batch], self.H.T)
        if self.normalize:
            raise ValueError("Not implemented")
        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if exclude_seen:
            scores_array[user_profile_batch.nonzero()] = -np.inf
        # rank items and mirror column to obtain a ranking in descending score
        #ranking = (-scores_array).argsort(axis=1)
        #ranking = np.fliplr(ranking)
        #ranking = ranking[:,0:n]
        ranking = np.zeros((scores_array.shape[0],n), dtype=np.int)
        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]
            relevant_items_partition = (-scores).argpartition(n)[0:n]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]
        return ranking

    def recommend(self, playlist_id_array, cutoff=None, remove_seen_flag=True,remove_top_pop_flag=False,remove_CustomItems_flag=False, export=False):

        #If is a scalar transform it in a 1-cell array
        if np.isscalar(playlist_id_array):
            playlist_id_array = np.atleast_1d(playlist_id_array)
            single_user = True
        else:
            single_user = False


        if cutoff==None:
            cutoff=self.URM_train.shape[1]-1
        scores = np.dot(self.W[playlist_id_array], self.H.T)

        for user_index in range(len(playlist_id_array)):
            user_id = playlist_id_array[user_index]
            if remove_seen_flag:
                scores[user_index, :] = self._remove_seen_on_scores(user_id, scores[user_index, :])

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

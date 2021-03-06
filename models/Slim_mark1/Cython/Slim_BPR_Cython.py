import subprocess
import os, sys
import numpy as np

from utils.OfflineDataLoader import OfflineDataLoader
from base.BaseRecommender import RecommenderSystem
from models.Slim_mark1.Slim_BPR import Slim_BPR_Recommender_Python
from models.Slim_mark1.Cython.Slim_BPR_Cython_Epoch import Slim_BPR_Cython_Epoch


class Slim_BPR_Recommender_Cython(Slim_BPR_Recommender_Python,RecommenderSystem):

    RECOMMENDER_NAME = "SLIM_BPR_Recommender_mark1"
    def __init__(self, URM_train, positive_threshold=1,
                 recompile_cython=False, sparse_weights=False,
                 symmetric=True, sgd_mode='adagrad'):

        super(Slim_BPR_Recommender_Cython, self).__init__(URM_train,
                                                          positive_threshold=positive_threshold,
                                                          sparse_weights=sparse_weights)
        self.sgd_mode = sgd_mode
        self.symmetric = symmetric
        self.parameters = None

        if not sparse_weights:
            num_items = URM_train.shape[1]
            requiredGB = 8 * num_items ** 2 / 1e+06
            if symmetric:
                requiredGB /= 2
            #print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} items is {:.2f} MB".format(num_items, requiredGB))

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def __str__(self):
        representation = "Slim BPR with Cython mark1"
        return representation

    def fit(self, epochs=50,
            URM_test=None,
            filterTopPop=False,
            minRatingsPerUser=1,
            batch_size=1000,
            validate_every_N_epochs=1,
            start_validation_after_N_epochs=0,
            lambda_i=1e-4,
            lambda_j=1e-4,
            learning_rate=0.020,
            topK=500,
            sgd_mode='adagrad',
            save_model = False,
            best_parameters=False,
            offline=True,submission=False):
        self.parameters = "positive_threshold= {0}, sparse_weights= {1}, symmetric= {2},sgd_mode= {3}, lambda_i={4}, " \
                          "lambda_j={5}, learning_rate={6}, topK={7}, epochs= {8}".format(
        self.positive_threshold,self.sparse_weights,self.symmetric,self.sgd_mode,lambda_i,lambda_j,learning_rate,topK,epochs)
        if offline:
            m = OfflineDataLoader()
            folder, file = m.get_model(self.RECOMMENDER_NAME, training=(not submission))
            self.loadModel(folder_path=folder,file_name=file)
        else:
            self.save_model = save_model
            # Select only positive interactions
            URM_train_positive = self.URM_train.copy()

            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
            URM_train_positive.eliminate_zeros()

            if best_parameters:
                m = OfflineDataLoader()
                folder_slim, file_slim = m.get_parameter(self.RECOMMENDER_NAME)
                self.loadModel(folder_path=folder_slim,file_name=file_slim)
                self.cythonEpoch = Slim_BPR_Cython_Epoch(
                    self.URM_mask,
                    sparse_weights=self.sparse_weights,
                    learning_rate=learning_rate,
                    batch_size=1,
                    symmetric=self.symmetric)
                result = super(Slim_BPR_Recommender_Cython, self).fit_alreadyInitialized(
                    epochs=epochs,
                    URM_test=URM_test,
                    filterTopPop=filterTopPop,
                    minRatingsPerUser=minRatingsPerUser,
                    batch_size=batch_size,
                    validate_every_N_epochs=validate_every_N_epochs,
                    start_validation_after_N_epochs=start_validation_after_N_epochs)

            else:
                self.sgd_mode = sgd_mode
                self.cythonEpoch = Slim_BPR_Cython_Epoch(
                    self.URM_mask,
                    sparse_weights=self.sparse_weights,
                    topK=topK,
                    learning_rate=learning_rate,
                    li_reg=lambda_i,
                    lj_reg=lambda_j,
                    batch_size=1,
                    symmetric=self.symmetric,
                    sgd_mode=sgd_mode)
                result = super(Slim_BPR_Recommender_Cython, self).fit_alreadyInitialized(
                    epochs=epochs,
                    URM_test=URM_test,
                    filterTopPop=filterTopPop,
                    minRatingsPerUser=minRatingsPerUser,
                    batch_size=batch_size,
                    validate_every_N_epochs=validate_every_N_epochs,
                    start_validation_after_N_epochs=start_validation_after_N_epochs,
                    lambda_i=lambda_i,
                    lambda_j=lambda_j,
                    learning_rate=learning_rate,
                    topK=topK)
                return result

        if save_model:
            self.saveModel("saved_models/submission/",file_name="SLIM_BPR_Recommender_mark1_submission_model")
        return self.W

    def runCompilationScript(self):
        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root
        compiledModuleSubfolder = "/models/Slim_mark1/Cython"
        # fileToCompile_list = ['Sparse_Matrix_CSR.pyx', 'SLIM_BPR_Cython_Epoch.pyx']
        fileToCompile_list = ['Slim_BPR_Cython_Epoch.pyx']
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

    def updateSimilarityMatrix(self):
        self.S = self.cythonEpoch.get_S()
        if self.sparse_weights:
            self.W_sparse = self.S
        else:
            self.W = self.S

    def epochIteration(self):
        self.cythonEpoch.epochIteration_Cython()


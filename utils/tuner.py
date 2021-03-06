#!/usr/bin/env python3
from models.Slim_ElasticNet.SlimElasticNetRecommender import SLIMElasticNetRecommender
from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender
from models.KNN.Item_KNN_CBFRecommender import ItemKNNCBFRecommender
from models.Slim_mark1.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython as Slim_mark1
from models.Slim_mark2.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython as Slim_mark2
from models.graph.P3AlphaRecommender import P3alphaRecommender
from models.graph.RP3BetaRecommender import RP3betaRecommender
from models.MF_mark2.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython
from models.MF_mark2.PureSVD import PureSVDRecommender
from models.offline.ItemTreeRecommender_offline import ItemTreeRecommender_offline
from models.offline.PartyRecommender_offline import PartyRecommender_offline
from models.offline.PyramidRecommender_offline import PyramidRecommender_offline
from models.offline.PyramidItemTreeRecommender_offline import PyramidItemTreeRecommender_offline
from models.offline.HybridEightRecommender_offline import HybridEightRecommender_offline
from models.offline.ComboRecommender_offline import ComboRecommender_offline
from models.offline.SingleNeuronRecommender_offline import SingleNeuronRecommender_offline
from models.FW_Similarity.CFWBoostingRecommender import CFWBoostingRecommender
from parameter_tuning.BayesianSearch import BayesianSearch
from parameter_tuning.AbstractClassSearch import DictionaryKeys
import traceback, pickle
from utils.PoolWithSubprocess import PoolWithSubprocess
from utils.OfflineDataLoader import OfflineDataLoader
import numpy as np


def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, n_cases, output_root_path,
                                            metric_to_optimize):
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10,15, 20,25,30,40, 50,60,75,80, 100,125, 150,175, 200,250, 300,350, 400,450, 500,550, 600,650, 700,750, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10,25, 50,75, 100,125,150,175, 200,250, 300,350,400,450, 500,750, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]
    if similarity_type in ["cosine", "asymmetric"]:
        hyperparamethers_range_dictionary["feature_weighting"] = ["none", "BM25", "TF-IDF"]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)


def run_KNNCBFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, ICM_train, n_cases,
                                             output_root_path, metric_to_optimize):
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    if similarity_type in ["cosine", "asymmetric"]:
        hyperparamethers_range_dictionary["feature_weighting"] = ["none", "BM25", "TF-IDF"]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)


def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, n_cases=50,
                               evaluator_validation=None, evaluator_test=None, metric_to_optimize="PRECISION",
                               output_root_path="tuned_parameters", parallelizeKNN=False):
    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)
    # Put each models to their own Directory
    rec_model_folder_path = output_root_path + "/" + recommender_class.RECOMMENDER_NAME + "/"
    if not os.path.exists(rec_model_folder_path):
        os.makedirs(rec_model_folder_path)
    output_root_path = rec_model_folder_path

    ##########################################################################################################
    this_output_root_path = output_root_path + recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)
    parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation,
                                     evaluator_test=evaluator_test)
    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    run_KNNCBFRecommender_on_similarity_type_partial = partial(
        run_KNNCBFRecommender_on_similarity_type,
        parameterSearch=parameterSearch,
        URM_train=URM_train,
        ICM_train=ICM_object,
        n_cases=n_cases,
        output_root_path=this_output_root_path,
        metric_to_optimize=metric_to_optimize)
    if parallelizeKNN:
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

    else:
        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)


def runParameterSearch_Collaborative(recommender_class, URM_train,ICM,W_sparse_CF,metric_to_optimize="PRECISION",
                                     evaluator_validation=None, evaluator_test=None,
                                     evaluator_validation_earlystopping=None,
                                     output_root_path="tuned_parameters", parallelizeKNN=True, n_cases=30):
    from parameter_tuning.AbstractClassSearch import DictionaryKeys
    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)
    # Put each models to their own Directory
    rec_model_folder_path = output_root_path + "/" + recommender_class.RECOMMENDER_NAME + "/"
    if not os.path.exists(rec_model_folder_path):
        os.makedirs(rec_model_folder_path)
    output_root_path = rec_model_folder_path

    try:
        output_root_path_rec_name = output_root_path + recommender_class.RECOMMENDER_NAME
        parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)
        ##########################################################################################################
        if recommender_class is UserKNNCFRecommender:
            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]
            run_KNNCFRecommender_on_similarity_type_partial = partial(
                run_KNNCFRecommender_on_similarity_type,
                parameterSearch=parameterSearch,
                URM_train=URM_train,
                n_cases=n_cases,
                output_root_path=output_root_path_rec_name,
                metric_to_optimize=metric_to_optimize)
            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)
            else:
                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)
            return
        ##########################################################################################################
        if recommender_class is ItemKNNCBFRecommender:
            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]
            run_KNNCFRecommender_on_similarity_type_partial = partial(
                run_KNNCBFRecommender_on_similarity_type,
                parameterSearch=parameterSearch,
                URM_train=URM_train,
                ICM_train=ICM,
                n_cases=n_cases,
                output_root_path=output_root_path_rec_name,
                metric_to_optimize=metric_to_optimize)
            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)
            else:
                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)
            return
        ##########################################################################################################

        if recommender_class is ItemKNNCFRecommender:
            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]
            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=n_cases,
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize)
            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:
                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)
            return

        ##########################################################################################################

        if recommender_class is P3alphaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10,15, 20,25,30,40, 50,60,75,80, 100,125, 150,175, 200,250, 300,350, 400,450, 500,550, 600,650, 700,750, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 3)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800,900,1000]
            hyperparamethers_range_dictionary["l1_ratio"] = [1e-1,1e-2, 1e-3, 1e-4,1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################
        if recommender_class is CFWBoostingRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["add_zeros_quota"] = range(0,1)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]



            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM, Slim_mark2],
                                    DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                    DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                    DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                    DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}                            

        ##########################################################################################################

        if recommender_class is RP3betaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10,15, 20,25,30,40, 50,60,75,80, 100,125, 150,175, 200,250, 300,350, 400,450, 500,550, 600,650, 700,750, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 3)
            hyperparamethers_range_dictionary["beta"] = range(0, 3)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["batch_size"] = [1]
            hyperparamethers_range_dictionary["positive_reg"] = [0.0,1e-1,1e-2, 1e-3,1e-4,1e-5, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["negative_reg"] = [0.0,1e-1,1e-2, 1e-3,1e-4,1e-5, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-2,2e-2,3e-2,4e-2,5e-2, 1e-3,2e-3,3e-3,4e-3,5e-3, 1e-4,2e-4,3e-4,4e-4,5e-4, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is PureSVDRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = list(range(0, 250, 5))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
        #########################################################################################################

        if recommender_class is Slim_mark1:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "sgd", "rmsprop"]
            hyperparamethers_range_dictionary["learning_rate"] = [0.1, 1e-2, 1e-3, 1e-4]
            hyperparamethers_range_dictionary["lambda_i"] = [ 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1e-1,1e-2,
                                                             1e-3,1e-4,1e-5, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["lambda_j"] = [ 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1e-1,1e-2,
                                                             1e-3,1e-4,1e-5, 1e-6, 1e-9]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'sparse_weights': False,
                                                                               'symmetric': True,
                                                                               'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
        #########################################################################################################
        if recommender_class is ItemTreeRecommender_offline:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["alpha"] = list(np.linspace(0.2, 0.85, 150))
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["beta"] = list(np.linspace(0.2, 0.85, 150))
            hyperparamethers_range_dictionary["gamma"] = list(np.linspace(0.2, 0.85, 150))
            hyperparamethers_range_dictionary["omega"] = list(np.linspace(0.2, 0.85, 150))
            hyperparamethers_range_dictionary["theta"] = list(np.linspace(0.2, 0.85, 150))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
        #########################################################################################################
        if recommender_class is PartyRecommender_offline:
            hyperparamethers_range_dictionary = {}

            # hyperparamethers_range_dictionary["alpha"] = list(np.linspace(0.2, 0.85, 150))
            # # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            # hyperparamethers_range_dictionary["beta"] = list(np.linspace(0.2, 0.85, 150))
            # hyperparamethers_range_dictionary["gamma"] = list(np.linspace(0.2, 0.85, 150))
            # hyperparamethers_range_dictionary["theta"] = list(np.linspace(0.2, 0.85, 150))
            # hyperparamethers_range_dictionary["omega"] = list(np.linspace(0.2, 0.85, 150))
            hyperparamethers_range_dictionary["alpha"] = range(0,1)
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["beta"] = range(0,1)
            hyperparamethers_range_dictionary["gamma"] = range(0,1)
            hyperparamethers_range_dictionary["theta"] = range(0,1)
            hyperparamethers_range_dictionary["omega"] = range(0,1)
            hyperparamethers_range_dictionary["coeff"] = range(1,50)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
        #########################################################################################################

        if recommender_class is PyramidRecommender_offline:
            hyperparamethers_range_dictionary = {}

            hyperparamethers_range_dictionary["alpha"] = range(0,3)
            hyperparamethers_range_dictionary["beta"] = range(0,3)
            hyperparamethers_range_dictionary["gamma"] = range(0,3)
            hyperparamethers_range_dictionary["chi"] = range(0,5)
            hyperparamethers_range_dictionary["psi"] = range(0,5)
            hyperparamethers_range_dictionary["omega"] = range(0,10)
            hyperparamethers_range_dictionary["coeff"] = range(1,40)


            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################
        if recommender_class is HybridEightRecommender_offline:
    
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["alpha"] = range(0, 4)
            hyperparamethers_range_dictionary["beta"] = range(0, 4)
            hyperparamethers_range_dictionary["gamma"] = range(0, 4)
            hyperparamethers_range_dictionary["delta"] = range(0, 4)
            hyperparamethers_range_dictionary["epsilon"] = range(0, 4)
            hyperparamethers_range_dictionary["zeta"] = range(0, 4)
            hyperparamethers_range_dictionary["eta"] = range(0, 4)
            hyperparamethers_range_dictionary["theta"] = range(0,4)
            hyperparamethers_range_dictionary["coeff"] = range(1, 50)
            hyperparamethers_range_dictionary["chi"] = range(0, 10)
            hyperparamethers_range_dictionary["psi"] = range(0, 10)
            hyperparamethers_range_dictionary["omega"] = range(0, 10)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train,ICM],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        if recommender_class is SingleNeuronRecommender_offline:
        
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["alpha"] = range(0, 20)
            hyperparamethers_range_dictionary["beta"] = range(0, 20)
            hyperparamethers_range_dictionary["gamma"] = range(0, 20)
            hyperparamethers_range_dictionary["delta"] = range(0, 20)
            hyperparamethers_range_dictionary["epsilon"] = range(0, 20)
            hyperparamethers_range_dictionary["zeta"] = range(0, 20)
            hyperparamethers_range_dictionary["eta"] = range(0, 20)
            hyperparamethers_range_dictionary["theta"] = range(0,20)
           


            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train,ICM],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################


        if recommender_class is ComboRecommender_offline:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["alpha"] = range(0, 20)
            hyperparamethers_range_dictionary["beta"] = range(0, 20)
            hyperparamethers_range_dictionary["gamma"] = range(0, 20)
            hyperparamethers_range_dictionary["theta"] = range(0, 20)
            hyperparamethers_range_dictionary["delta"] = range(0, 20)
            hyperparamethers_range_dictionary["epsilon"] = range(0, 20)


            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        if recommender_class is PyramidItemTreeRecommender_offline:
            hyperparamethers_range_dictionary = {}

            hyperparamethers_range_dictionary["alpha"] = range(0,5)
            hyperparamethers_range_dictionary["beta"] = range(0,5)
            hyperparamethers_range_dictionary["gamma"] = range(0,5)
            hyperparamethers_range_dictionary["sigma"] = range(0,5)
            hyperparamethers_range_dictionary["tau"] = range(0,5)
            hyperparamethers_range_dictionary["chi"] = range(0,5)
            hyperparamethers_range_dictionary["psi"] = range(0,5)
            hyperparamethers_range_dictionary["omega"] = range(0,20)
            hyperparamethers_range_dictionary["coeff"] = range(1,50)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train,ICM],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
                                     
        #########################################################################################################
        if recommender_class is Slim_mark2:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            hyperparamethers_range_dictionary["lambda_i"] = [0.0,1e-1,1e-2, 1e-3,1e-4,1e-5, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["lambda_j"] = [0.0,1e-1,1e-2, 1e-3,1e-4,1e-5, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [0.01,0.001,0.005,0.025,0.0025]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights': True,
                                                                               'symmetric': True,
                                                                               'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 10,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################


        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 n_cases=n_cases,
                                                 output_root_path=output_root_path_rec_name,
                                                 metric=metric_to_optimize)

    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_root_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()


import os, multiprocessing
from functools import partial
from data.PlaylistDataReader import PlaylistDataReader
from utils.config import clear


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """
    clear()
    dataReader = PlaylistDataReader()
    dataReader.generate_datasets()
    URM_train = dataReader.get_URM_train()
    # URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    output_root_path = "tuned_parameters"
    m = OfflineDataLoader()
    fold,fil = m.get_model(ItemKNNCFRecommender.RECOMMENDER_NAME,training=True)
    m1 = ItemKNNCFRecommender(URM_train,ICM)
    m1.loadModel(folder_path=fold,file_name=fil)
    W_sparse_CF = m1.W_sparse

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    collaborative_algorithm_list = [
        #P3alphaRecommender,
        #RP3betaRecommender,
        #ItemKNNCFRecommender,
        #UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # Slim_mark1,
        # Slim_mark2,
        # ItemTreeRecommender_offline
        # SLIMElasticNetRecommender,
       # PartyRecommender_offline
       # PyramidRecommender_offline
       #  ItemKNNCBFRecommender
       # PyramidItemTreeRecommender_offline
        #HybridEightRecommender_offline
        #ComboRecommender_offline
        SingleNeuronRecommender_offline
     # CFWBoostingRecommender

    ]

    from parameter_tuning.AbstractClassSearch import EvaluatorWrapper
    from base.evaluation.Evaluator import SequentialEvaluator

    evaluator_validation_earlystopping = SequentialEvaluator(URM_test, cutoff_list=[10])
    evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])

    evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       ICM = ICM,
                                                       W_sparse_CF = W_sparse_CF,
                                                       metric_to_optimize="MAP",
                                                       evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=evaluator_test,
                                                       n_cases=250,
                                                       output_root_path=output_root_path)

    for recommender_class in collaborative_algorithm_list:
        try:
            runParameterSearch_Collaborative_partial(recommender_class)
        except Exception as e:
            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()


if __name__ == '__main__':
    read_data_split_and_search()

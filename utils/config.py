import json
from bunch import Bunch
import os
from subprocess import call
#
from models.Slim_BPR.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython
from models.KNN.Item_KNN_CBFRecommender import ItemKNNCBFRecommender
from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender
from models.Slim_BPR.Slim_BPR import Slim_BPR_Recommender_Python
from models.MatrixFactorization.FunkSVD import FunkSVD
from models.MatrixFactorization.AsymmetricSVD import AsySVD
from models.MatrixFactorization.IALS import IALS_numpy
from models.MatrixFactorization.BPRMF import BPRMF
from models.MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython

# define clear function
def clear():
    # check and make call for specific operating system
    _ = call('clear' if os.name == 'posix' else 'cls')


class Configurator(object):
    def __init__(self, jsonFile):
        self.dataFile = jsonFile
        self.configs = self.process_config(self.dataFile)

    @staticmethod
    def get_config_from_json(json_file):
        """
        Get the config from a json file
        :param json_file:
        :return: config(namespace) or config(dictionary)
        """
        # parse the configurations from the config json file provided
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)

        # convert the dictionary to a namespace using bunch lib
        config = Bunch(config_dict)

        return config, config_dict

    def process_config(self, json_file):
        configs, _ = self.get_config_from_json(json_file)
        # config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/")
        # config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
        return configs

    def extract_models(self, dataReader):
        print("Configurator: The models are being extracted from the config file")
        recsys = list()
        models = list(self.configs.models)
        for model in models:
            if model["model_name"] == "user_knn_cf":
                recsys.append(UserKNNCFRecommender(dataReader.get_URM_train(),
                                                   dataReader.get_UCM(),
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"],
                                                   normalize=model["normalize"]))
            elif model["model_name"] == "item_knn_cf":
                recsys.append(ItemKNNCFRecommender(dataReader.get_URM_train(),
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"],
                                                   normalize=model["normalize"]))
            elif model["model_name"] == "item_knn_cbf":
                recsys.append(ItemKNNCBFRecommender(dataReader.get_URM_train(),
                                                    dataReader.get_ICM(),
                                                    sparse_weights=model["sparse_weights"],
                                                    verbose=model["verbose"],
                                                    similarity_mode=model["similarity_mode"]))
            elif model["model_name"] == "slim_bpr_python":
                recsys.append(Slim_BPR_Recommender_Python(dataReader.get_URM_train(),
                                                          positive_threshold=model["positive_threshold"],
                                                          sparse_weights= model["sparse_weights"]
                                                          ))
            elif model["model_name"] == "slim_bpr_cython":
                recsys.append(Slim_BPR_Recommender_Cython(dataReader.get_URM_train(),
                                                          positive_threshold=model["positive_threshold"],
                                                          recompile_cython=model["recompile_cython"],
                                                          sparse_weights= model["sparse_weights"],
                                                          symmetric= model["symmetric"],
                                                          sgd_mode= model["sgd_mode"]
                                                          ))
            elif model["model_name"] == "funksvd":
                recsys.append(FunkSVD(dataReader.get_URM_train()))

            elif model["model_name"] == "asysvd":
                recsys.append(AsySVD(dataReader.get_URM_train()))

            elif model["model_name"] == "mf_bpr_cython":
                recsys.append(MF_BPR_Cython(dataReader.get_URM_train(),
                                            recompile_cython=model["recompile_cython"]))
            elif model["model_name"] == "ials_numpy":
                recsys.append(IALS_numpy())
            elif model["model_name"] == "bprmf":
                recsys.append(BPRMF())
        print("Configurator: Models are extracted")

        return recsys

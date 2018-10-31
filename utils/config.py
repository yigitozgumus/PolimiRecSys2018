import json
from bunch import Bunch
import os
from subprocess import call

from models.Slim_BPR.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython
from models.KNN.Item_KNN_CBFRecommender import ItemKNNCBFRecommender
from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender
from models.Slim_BPR.Slim_BPR import Slim_BPR_Recommender_Python


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
        print("The models are being extracted from the config file")
        recsys = list()
        models = list(self.configs.models)
        for model in models:
            if model["model_name"] == "user_knn_cf":
                recsys.append(UserKNNCFRecommender(dataReader.URM_train,
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"]))
            elif model["model_name"] == "item_knn_cf":
                recsys.append(ItemKNNCFRecommender(dataReader.URM_train,
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"]))
            elif model["model_name"] == "item_knn_cbf":
                recsys.append(ItemKNNCBFRecommender(dataReader.URM_train,
                                                    dataReader.trainData,
                                                    dataReader.trackData,
                                                    sparse_weights=model["sparse_weights"],
                                                    verbose=model["verbose"],
                                                    similarity_mode=model["similarity_mode"],
                                                    useTrackPopularity=model["useTrackPopularity"],
                                                    useAlbumPopularity=model["useAlbumPopularity"],
                                                    useArtistPopularity=model["useArtistPopularity"],
                                                    normalizeFeatures=model["normalize"]))
            elif model["model_name"] == "slim_bpr_python":
                recsys.append(Slim_BPR_Recommender_Python(dataReader.URM_train,
                                                          positive_threshold=model["positive_threshold"],
                                                          sparse_weights= model["sparse_weights"]
                                                          ))
            elif model["model_name"] == "slim_bpr_cython":
                recsys.append(Slim_BPR_Recommender_Cython(dataReader.URM_train,
                                                          positive_threshold=model["positive_threshold"],
                                                          recompile_cython=model["recompile_cython"],
                                                          sparse_weights= model["sparse_weights"],
                                                          symmetric= model["symmetric"],
                                                          sgd_mode= model["sgd_mode"]
                                                          ))
        return recsys

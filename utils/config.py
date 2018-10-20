import json
from bunch import Bunch
import os
import pandas as pd
from subprocess import call
from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender

# define clear function
def clear():
    # check and make call for specific operating system
    _ = call('clear' if os.name == 'posix' else 'cls')

class Configurator(object):
    def __init__(self,jsonFile,dataReader):
        self.dataFile = jsonFile
        self.configs = self.process_config(self.dataFile)
        self.dataReader = dataReader

    def get_config_from_json(self,json_file):
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


    def process_config(self,json_file):
        configs, _ = self.get_config_from_json(json_file)
        # config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/")
        # config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
        return configs

    def extractModels(self):
        print("The models are being extracted from the config file")
        recsys = list()
        models = list(self.configs.models)
        for model in models:
            if model["model_name"] == "user_knn_cf":
                recsys.append(UserKNNCFRecommender(self.dataReader.URM_train,
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"]))
            elif model["model_name"] == "item_knn_cf":
                recsys.append(ItemKNNCFRecommender(self.dataReader.URM_train,
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"]))
        return recsys



        


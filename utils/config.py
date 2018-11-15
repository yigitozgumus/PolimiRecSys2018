import json
from bunch import Bunch
import os
from subprocess import call

from models.hybrid.SeqRandRecommender import SeqRandRecommender
from models.Slim_BPR.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython
from models.KNN.Item_KNN_CBFRecommender import ItemKNNCBFRecommender
from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender
from models.hybrid.UserItemAverage_Recommender import UserItemAvgRecommender
from models.Slim_BPR.Slim_BPR import Slim_BPR_Recommender_Python
from models.MatrixFactorization.FunkSVD import FunkSVD
from models.MatrixFactorization.AsymmetricSVD import AsySVD
from models.MatrixFactorization.IALS import IALS_numpy
from models.MatrixFactorization.BPRMF import BPRMF
from models.MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
from models.hybrid.TwoLevel_Hybrid_Recommender import TwoLevelHybridRecommender
from models.hybrid.ItemTreeRecommender import ItemTreeRecommender
from models.hybrid.SeqRandRecommender_mark2 import SeqRandRecommender_mark2
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

    def extract_models(self, dataReader,submission = False):
        print("Configurator: The models are being extracted from the config file")
        recsys = list()
        models = list(self.configs.models)
        data = dataReader.get_URM_train()
        if submission:
            data = dataReader.get_URM_all()
        for model in models:
            if model["model_name"] == "user_knn_cf":
                recsys.append(UserKNNCFRecommender(data,
                                                   dataReader.get_UCM(),
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"],
                                                   normalize=model["normalize"]))
            elif model["model_name"] == "item_knn_cf":
                recsys.append(ItemKNNCFRecommender(data,
                                                   sparse_weights=model["sparse_weights"],
                                                   verbose=model["verbose"],
                                                   similarity_mode=model["similarity_mode"],
                                                   normalize=model["normalize"]))
            elif model["model_name"] == "item_knn_cbf":
                recsys.append(ItemKNNCBFRecommender(data,
                                                    dataReader.get_ICM(),
                                                    sparse_weights=model["sparse_weights"],
                                                    verbose=model["verbose"],
                                                    similarity_mode=model["similarity_mode"]))
            elif model["model_name"] == "slim_bpr_python":
                recsys.append(Slim_BPR_Recommender_Python(data,
                                                          positive_threshold=model["positive_threshold"],
                                                          sparse_weights= model["sparse_weights"]
                                                          ))
            elif model["model_name"] == "slim_bpr_cython":
                recsys.append(Slim_BPR_Recommender_Cython(data,
                                                          positive_threshold=model["positive_threshold"],
                                                          recompile_cython=model["recompile_cython"],
                                                          sparse_weights= model["sparse_weights"],
                                                          symmetric= model["symmetric"],
                                                          sgd_mode= model["sgd_mode"]
                                                          ))
            elif model["model_name"] == "funksvd":
                recsys.append(FunkSVD(data))

            elif model["model_name"] == "asysvd":
                recsys.append(AsySVD(data))

            elif model["model_name"] == "mf_bpr_cython":
                recsys.append(MF_BPR_Cython(data,
                                            recompile_cython=model["recompile_cython"]))
            elif model["model_name"] == "ials_numpy":
                recsys.append(IALS_numpy())
            elif model["model_name"] == "bprmf":
                recsys.append(BPRMF())
            elif model["model_name"] == "user_item_avg":
                recsys.append(UserItemAvgRecommender(data,
                                                      dataReader.get_UCM(),
                                                      dataReader.get_ICM(),
                                                      sparse_weights=model["sparse_weights"],
                                                      verbose=model["verbose"],
                                                      similarity_mode=model["similarity_mode"],
                                                      normalize=model["normalize"],
                                                      alpha= model["alpha"]
                                                      ))
            elif model["model_name"] == "2levelhybrid":
                recsys.append(TwoLevelHybridRecommender(data,
                                                        dataReader.get_UCM(),
                                                        dataReader.get_ICM(),
                                                        sparse_weights=model["sparse_weights"],
                                                        verbose=model["verbose"],
                                                        similarity_mode=model["similarity_mode"],
                                                        normalize=model["normalize"],
                                                        alpha=model["alpha"],
                                                        avg=model["avg"]
                                                        ))
            elif model["model_name"] == "seqrand":
                recsys.append(SeqRandRecommender(data,
                                                 dataReader.get_URM_train_tfidf(),
                                                 dataReader.get_UCM(),
                                                 dataReader.get_ICM(),
                                                 dataReader.get_target_playlists_seq(),
                                                 sparse_weights=model["sparse_weights"],
                                                 verbose=model["verbose"],
                                                 similarity_mode=model["similarity_mode"],
                                                 normalize=model["normalize"],
                                                 alpha=model["alpha"],
                                                 beta=model["beta"],
                                                 gamma = model["gamma"]
                                                 ))
            elif model["model_name"] == "itemtree":
                recsys.append(ItemTreeRecommender(data,
                                                  dataReader.get_URM_train_tfidf(),
                                                  dataReader.get_ICM(),
                                                  sparse_weights=model["sparse_weights"],
                                                  verbose=model["verbose"],
                                                  similarity_mode=model["similarity_mode"],
                                                  normalize=model["normalize"],
                                                  alpha=model["alpha"],
                                                  beta=model["beta"],
                                                  gamma=model["gamma"]))
            elif model["model_name"] == "seqrand_mark2":
                recsys.append(SeqRandRecommender_mark2(data,
                                                 dataReader.get_URM_train_tfidf(),
                                                 dataReader.get_UCM(),
                                                 dataReader.get_ICM(),
                                                 dataReader.get_target_playlists_seq(),
                                                 sparse_weights=model["sparse_weights"],
                                                 verbose=model["verbose"],
                                                 similarity_mode=model["similarity_mode"],
                                                 normalize=model["normalize"],
                                                 alpha=model["alpha"],
                                                 beta=model["beta"],
                                                 gamma = model["gamma"],
                                                 omega = model["omega"]
                                                 ))
        print("Configurator: Models are extracted")

        return recsys

from models.Slim_ElasticNet.SlimElasticNetRecommender import SLIMElasticNetRecommender
from data.PlaylistDataReader import PlaylistDataReader
from utils.logger import Logger
from utils.config import clear, Configurator
import argparse
from utils.OfflineDataLoader import OfflineDataLoader
from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender
from models.KNN.Item_KNN_CBFRecommender import ItemKNNCBFRecommender
from models.graph.P3AlphaRecommender import P3alphaRecommender
from models.graph.RP3BetaRecommender import RP3betaRecommender
from models.MF_mark2.PureSVD import PureSVDRecommender
from models.Slim_mark1.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython as Slim_mark1
from models.Slim_mark2.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython as Slim_mark2
from models.offline.ItemTreeRecommender_offline import ItemTreeRecommender_offline
from models.offline.PartyRecommender_offline import PartyRecommender_offline
from models.offline.SingleNeuronRecommender_offline import SingleNeuronRecommender_offline
from utils.config import clear


from contextlib import contextmanager
import os
import json
import re

# function for changing directory
@contextmanager
def working_directory(directory):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)


def getModelName(file):
    lst = file.split("/")
    return lst[1]


def getSimName(file, saved=False):
    if saved:
        index = -3
    else:
        index = -4
    simList = ["asymmetric", "tversky", "cosine", "jaccard", "dice"]
    lst = file.split("_")
    if lst[index] in simList:
        return lst[index]
    else:
        return "default"


def getUniqueModelList(fileList):
    models = [getModelName(i) for i in fileList]
    return list(set(models))


def packageWithModel(fileList, saved=False):
    model_file = []
    for file in fileList:
        added = (getModelName(file), file, getSimName(file, saved))
        model_file.append(added)
    return model_file

def printOutMapValues(modelList, URM, ICM, modelsSoFar):
    map_dict = {i: dict() for i in modelsSoFar}
    m = OfflineDataLoader()
    for model in modelList:
        folder = str("/".join(model[1].split("/")[:-1]) + "/")
        file = model[1].split("/")[-1]
        if model[0] == "UserKNNCFRecommender":
            mod = UserKNNCFRecommender(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
           # print(model[0], model[2], mod.MAP)
        elif model[0] == "ItemKNNCFRecommender":
            mod = ItemKNNCFRecommender(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
           # print(model[0], model[2], mod.MAP)
        elif model[0] == "ItemKNNCBFRecommender":
             mod = ItemKNNCBFRecommender(URM, ICM)
             mod.loadModel(folder_path=folder, file_name=file, verbose=False)
             map_dict[model[0]][model[2]] = mod.MAP
           # print(model[0], model[2], mod.MAP)
        elif model[0] == "SLIM_BPR_Recommender_mark1":
            mod = Slim_mark1(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
           # print(model[0], model[2], mod.MAP)
        elif model[0] == "RP3_Beta_Recommender":
            mod = RP3betaRecommender(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
           # print(model[0], model[2], mod.MAP)
        elif model[0] == "P3_Alpha_Recommender":
            mod = P3alphaRecommender(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
           # print(model[0], model[2], mod.MAP)
        elif model[0] == "PureSVD":
            mod = PureSVDRecommender(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
           # print(model[0], model[2], mod.MAP)
        elif model[0] == "Slim_Elastic_Net_Recommender":
            mod = SLIMElasticNetRecommender(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
            #print(model[0], model[2], mod.MAP)
        elif model[0] == "SLIM_BPR_Recommender_mark2":
            mod = Slim_mark2(URM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
            #print(model[0], model[2], mod.MAP)
        # elif model[0] == "ItemTreeRecommender_offline":
        #     mod = ItemTreeRecommender_offline(URM,ICM)
        #     mod.loadModel(folder_path=folder, file_name=file, verbose=False)
        #     map_dict[model[0]][model[2]] = mod.MAP
            #print(model[0], model[2], mod.MAP)
        # elif model[0] == "PartyRecommender_offline":
        #     mod = PartyRecommender_offline(URM)
        #     mod.loadModel(folder_path=folder, file_name=file, verbose=False)
        #     map_dict[model[0]][model[2]] = mod.MAP
        #     #print(model[0], model[2], mod.MAP)
        elif model[0] == "SingleNeuronRecommender_offline":
            mod = SingleNeuronRecommender_offline(URM,ICM)
            mod.loadModel(folder_path=folder, file_name=file, verbose=False)
            map_dict[model[0]][model[2]] = mod.MAP
            #print(model[0], model[2], mod.MAP)

    return map_dict


def create_mapMax(model_dict):
    model_map_list = []
    import operator
    for model in sorted(list(model_dict.keys())):
        sorted_by_value = sorted(model_dict[model].items(), key=lambda kv: kv[1],reverse=True)
        if len(sorted_by_value) != 0:
            model_map_list.append((model,(sorted_by_value[0])))
    return model_map_list




def main():
    clear()
    data_reader = PlaylistDataReader()
    data_reader.build_URM()
    data_reader.build_ICM()
    filter = re.compile(r'\..+|.+\.txt$')
    parameterFolder = "tuned_parameters"
    listOfFolders = os.listdir(parameterFolder)
    filteredDirPaths = [parameterFolder + "/" + i for i in listOfFolders if not filter.search(i)]
    saved_parameterFolder = "saved_parameters"
    listofParameters = os.listdir(saved_parameterFolder)
    filteredSavedParameters = [saved_parameterFolder + "/" + i for i in listofParameters if not filter.search(i)]

    best_models = []
    best_model_parameters = []
    best_results = []
    files = []

    # get all the files
    for folder in filteredDirPaths:
        with working_directory(folder):
            filePaths = [folder + "/" + i for i in os.listdir(".")]
            files.extend(filePaths)

    # Define error filter
    errorFilter = re.compile(r'Error.+')
    # Make it error free
    errorFilteredFiles = [i for i in files if not errorFilter.search(i)]
    bestModelFilter = re.compile(r'best_model$')
    modelFiles = [i for i in files if bestModelFilter.search(i)]
    parameterFilter = re.compile(r'best_parameters$')
    parameterFiles = [i for i in files if parameterFilter.search(i)]
    resultFilter = re.compile(r'best_result_test$')
    resultFiles = [i for i in files if resultFilter.search(i)]
    modelFiles_t = packageWithModel(modelFiles)
    parameterFiles_t = packageWithModel(parameterFiles)
    resultFiles_t = packageWithModel(resultFiles)
    modelsSoFar = getUniqueModelList(modelFiles)

    a = printOutMapValues(resultFiles_t, data_reader.URM_all, data_reader.ICM, modelsSoFar)

    print("Models with their MAP Values\n")
    print("###################################################")
    print(json.dumps(a, sort_keys=True, indent=4))
    print("###################################################\n")
    print("Maximum MAP for each model\n")
    print("###################################################")
    b = create_mapMax(a)
    for i in b:
        print(i)



if __name__ == "__main__":
    main()

from utils.util import working_directory
import os
import re
import models as m

class OfflineDataLoader(object):
    def __init__(self,model_folder="saved_models",parameter_folder="saved_parameters"):
        super(OfflineDataLoader, self).__init__()
        self.repository = "tuned_parameters"
        self.model_folder = model_folder
        self.parameter_folder = parameter_folder
        self.training = self.model_folder + "/" + "training"
        self.submission = self.model_folder + "/" + "submission"
        self.training_models = self.get_models(self.training)
        self.submission_models = self.get_models(self.submission)
        self.parameter_files = self.get_models(self.parameter_folder)
        self.repository_files = self.build_repository(self.repository)

    def get_model(self,model_name,training=True):
        if training:
            result = [i for i in self.training_models if re.compile(model_name).search(i)]
            folder_path = str("/".join(result[0].split("/")[:-1])+"/")
            file_name = result[0].split("/")[-1]
            return folder_path,file_name
        else:
            result = [i for i in self.submission_models if re.compile(model_name).search(i)]
            folder_path = str("/".join(result[0].split("/")[:-1])+"/")
            file_name = result[0].split("/")[-1]
            return folder_path,file_name

    def get_parameter(self,model_name):
        result = [i for i in self.parameter_files if re.compile(model_name).search(i)]
        folder_path = str("/".join(result[0].split("/")[:-1])+"/")
        file_name = result[0].split("/")[-1]
        return folder_path,file_name

    def get_models(self,folder_name):
        fileList = os.listdir(folder_name)
        filter = re.compile(r'\..+|.+\.txt$')
        filtered_files = [folder_name + "/" + i for i in fileList if not filter.search(i)]
        return filtered_files

    def build_repository(self,repo_folder):
        filter = re.compile(r'\..+|.+\.txt$')
        listOfFolders = os.listdir(repo_folder)
        filteredDirPaths = [repo_folder+"/"+i for i in listOfFolders if not filter.search(i)]
        files = []
        for folder in filteredDirPaths:
            with working_directory(folder):
                filePaths = [folder +"/"+ i for i in os.listdir(".")]
                files.extend(filePaths)
        # Categorize
        # Define error filter
        errorFilter = re.compile(r'Error.+')
        # Make it error free
        errorFilteredFiles = [i for i in files if not errorFilter.search(i)]
        bestModelFilter = re.compile(r'best_model$')
        self.best_models = [i for i in files if bestModelFilter.search(i)]
        parameterFilter = re.compile(r'best_parameters$')
        self.best_parameters = [i for i in files if parameterFilter.search(i)]
        resultFilter = re.compile(r'best_result_test$')
        self.best_results = [i for i in files if resultFilter.search(i)]

        



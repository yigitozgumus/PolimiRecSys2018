from time import asctime
from pathlib import Path
import os 
class Logger(object):

    def __init__(self, data, logFile, verbose=False):
        super(Logger, self).__init__()
        self.verbose = verbose
        self.logFile = logFile
        self.data = data
        self.submission_list = []
        self.time = asctime()
        self.dir = "Logs/"

    def export_experiments(self, model,results):
        index = len(list(os.listdir("./experiments_2"))) 
        self.submission_list = []
        filePath = "experiments_2/exp-"+str(index) + ".csv"
        fileName = "exp-" + str(index) + ".csv"
        self.submission_list.append((fileName,filePath,self.time,model, model.parameters,results))
        f = open(filePath,"w+")
        f.write("playlist_id,track_ids\n")
        for ind, playlist_id in enumerate(self.data['playlist_id']):
            f.write(str(playlist_id) + ',' + str(model.recommend(playlist_id, cutoff=10, export=True)).replace(",", "") + '\n')
        f.close()
        print("Logger: The experiment {0} is exported to the {1}".format(fileName,filePath))
        index+=1

    def export_single_submission(data, model):
        f = open("experiments/exp-" + asctime() + ".csv", "w+")
        f.write("playlist_id,track_ids\n")
        for ind, playlist_id in enumerate(data['playlist_id']):
            f.write(str(playlist_id) + ',' + model.recommend(playlist_id, cutoff=10, export=True) + '\n');
        f.close()

    def createLogFile(self,filename):
        f = open(self.dir + filename,"w+")
        f.write("# Experiment Logs\n")
        f.write(
            "| Experiment Name | Date | Model Name | Parameters |Results | Submission|\n")
        f.write(
            "|---              |---   |---         |---         |---     |---        |")
        f.close()

    def log_experiment(self,submission=False):
        submit = ""
        if submission:
            submit = "Full URM"
        if self.logFile is None:
            f = open(self.dir + "Logs.md", "a")
        else:
            logFile = Path(self.dir + self.logFile)
            if logFile.is_file():
                f = open(self.dir + self.logFile, "a")
            else:
                self.createLogFile(self.logFile)
                f = open(self.dir + self.logFile, "a")
        for submission in self.submission_list:
            f.write("\n|[" + submission[0] + "](" + submission[1]+ ")|" + "|".join(list(map(str,submission[2:]))) + "|" + submit + "|")

        f.close()

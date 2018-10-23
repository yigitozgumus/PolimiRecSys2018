from time import asctime

import os 
class Logger(object):

    def __init__(self, data, verbose=False):
        super(Logger, self).__init__()
        self.verbose = verbose
        self.data = data
        self.submission_list = []
        self.time = asctime()

    def export_experiments(self, model_bundle):
        index = len(list(os.listdir("./experiments"))) -2
        for model in model_bundle:
            filePath = "experiments/exp-"+str(index) + ".csv"
            fileName = "exp-" + str(index) + ".csv"
            self.submission_list.append((fileName,filePath,self.time,model, model.map,model.precision,model.recall,model.parameters))
            f = open(filePath,"w+")
            f.write("playlist_id,track_ids\n")
            for ind, playlist_id in enumerate(self.data['playlist_id']):
                f.write(str(playlist_id) + ',' + model.recommend(playlist_id, n=10,export=True) + '\n');
            f.close()
            index+=1

    def export_single_submission(data, model):
        f = open("experiments/exp-" + asctime() + ".csv", "w+")
        f.write("playlist_id,track_ids\n")
        for ind, playlist_id in enumerate(data['playlist_id']):
            f.write(str(playlist_id) + ',' + model.recommend(playlist_id, n=10,export=True) + '\n');
        f.close()

    def log_experiment(self):
        f = open("Logs.md", "a")
        for submission in self.submission_list:
            f.write("\n|[" + submission[0] + "](" + submission[1]+ ")|" + \
                             submission[2] + "|" +\
                             str(submission[3]) + "|" + \
                             submission[4] + "|" + \
                             submission[5]+ "|" + \
                             submission[6] + "|" + \
                             submission[7] + "||" )
        f.close()

from time import asctime

import os 
class Logger(object):

    def __init__(self, data, verbose=False):
        super(Logger, self).__init__()
        self.verbose = verbose
        self.data = data
        self.submission_list = []
        self.time = asctime().replace(" ", "_")

    def export_experiments(self, model_bundle):
        index = len(list(os.listdir("./experiments"))) -2
        for model in model_bundle:
            fileName = "experiments/exp-"+str(index)+"-" + self.time + ".csv"
            self.submission_list.append((fileName, model))
            f = open(fileName,"w+")
            f.write("playlist_id,track_ids\n")
            for ind, playlist_id in enumerate(self.data['playlist_id']):
                f.write(str(playlist_id) + ',' + model.recommend(playlist_id, n=10,export=True) + '\n');
            f.close()
            index+=1

    def export_single_submission(data, model):
        f = open("experiments/exp-" + asctime() + ".csv", "w+")
        f.write("playlist_id,track_ids\n")
        for ind, playlist_id in enumerate(data['playlist_id']):
            f.write(str(playlist_id) + ',' + model.recommend(playlist_id, n=10) + '\n');
        f.close()

    def log_experiment(self):
        f = open("Logs.txt", "a")
        for submission in self.submission_list:
            f.write("\n" + submission[0][12:] + ", " + str(submission[1]))
        f.close()

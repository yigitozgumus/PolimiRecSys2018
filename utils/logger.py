from time import asctime


class Logger(object):

    def __init__(self, data, verbose=False):
        super(Logger, self).__init__()
        self.verbose = verbose
        self.data = data
        self.submission_list = []
        self.time = asctime()

    def export_submissions(self, model_bundle):
        for model in model_bundle:
            fileName = "submissions/submission-" + self.time + ".csv"
            self.submission_list.append((fileName, model))
            f = open(fileName,"w+")
            f.write("playlist_id,track_ids\n")
            for ind, playlist_id in enumerate(self.data['playlist_id']):
                f.write(str(playlist_id) + ',' + model.recommend(playlist_id, n=10) + '\n');
            f.close()

    def export_single_submission(data, model):
        f = open("submissions/submission-" + asctime() + ".csv", "w+")
        f.write("playlist_id,track_ids\n")
        for ind, playlist_id in enumerate(data['playlist_id']):
            f.write(str(playlist_id) + ',' + model.recommend(playlist_id, n=10) + '\n');
        f.close()

    def log_experiment(self):
        f = open("Logs", "a")
        for submission in self.submission_list:
            f.write("\n" + submission[0] + ", " + str(submission[1]))
        f.close()

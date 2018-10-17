import numpy as np
import time
import pandas as pd
from base.Metrics import Metrics
from base.RecommenderUtils import check_matrix


class RecommenderSystem(object):

  
    def get_user_relevant_items(self, user_id):

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def get_user_test_ratings(self, user_id):

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def evaluate(self, at=10):
        start_time = time.time()
        print("evaluation has started")
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0
        self.URM_test = check_matrix(self.URM_test,format="csr")
        num_eval = 0
       # self.playlists = self.playlists['playlist_id'].values.reshape(len(self.playlists['playlist_id']), 1)
        metric = Metrics()
        for playlist_id in self.playlists:

            relevant_items = self.get_user_relevant_items(playlist_id)

        if len(relevant_items) > 0:
            recommended_items = self.model.recommend(playlist_id, at=at)
            num_eval += 1
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
            cumulative_precision += metric.precision(is_relevant)
            cumulative_recall += metric.recall(is_relevant,relevant_items)
            cumulative_MAP += metric.map(is_relevant,relevant_items)

            if num_eval % 100 == 0 or num_eval == len(self.playlists) - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                    num_eval,
                    100.0 * float(num_eval + 1) / len(self.playlists),
                    time.time() - start_time,
                    float(num_eval) / (time.time() - start_time)))

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))


    def pipeline(self):
       # self.preprocess()
        self.model.fit()
        self.evaluate()



def run():
    pass


if __name__ == "__run__":
    run()

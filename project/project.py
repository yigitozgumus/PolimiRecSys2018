import numpy as np
import sklearn as sk
import pandas as pd
import scipy as sp
import seaborn as sns


class RecommenderSystem(object):
    
    def __init__(self, config):
        self.config = config
        self.train_data = pd.read_csv(config.data_files[0])
        self.track_data = pd.read_csv(config.data_files[1])
        self.target = pd.read_csv(config.data_files[2])
        self.train_split,self.test_split =  self.preprocess()

    def preprocess(self):
            return 1,1

    def precision(self,recommended_items,relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
        return precision_score

    def recall(self, recommended_items,relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
        return recall_score

    def meanAveragePrecision(self,recommended_items,relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
        return map_score

    def evaluate(self,id_set, model,at=10):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        for playlist_id in id_set:

            relevant_items = self.test_data[playlist_id].indices

            if len(relevant_items) > 0:
                recommended_items = model.recommend(playlist_id, at=at)
                num_eval += 1

                cumulative_precision += self.precision(recommended_items, relevant_items)
                cumulative_recall += self.recall(recommended_items, relevant_items)
                cumulative_MAP += self.meanAveragePrecision(recommended_items, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))
    

    def pipeline(self,model):
        pass


def run():
    pass


if __name__ == "__run__":
    run()

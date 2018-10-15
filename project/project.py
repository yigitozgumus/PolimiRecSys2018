import numpy as np
import pandas as pd
import scipy.sparse as sps
from utils.config import import_data_file


class RecommenderSystem(object):

    def __init__(self, config, model):
        self.config = config
        self.train_data = import_data_file(config.data_files[0])
        self.track_data = import_data_file(config.data_files[1])
        self.playlists = None
        self.urm_train, self.urm_test = self.preprocess(0.8)
        self.model = model(self.urm_train)

    def preprocess(self, split):
        interactionColumn = np.array(self.train_data['track_id']).flatten()
        playlistColumn = np.array(self.train_data['playlist_id']).flatten()
        trackColumn = np.array(self.train_data['track_id']).flatten()
        URM_all = sps.coo_matrix((interactionColumn, (playlistColumn, trackColumn)))
        URM_all.tocsr()
        numInteractions = URM_all.nnz
        train_mask = np.random.choice([True, False], numInteractions, p=[split, 1 - split])

        URM_train = sps.coo_matrix(
            (interactionColumn[train_mask], (playlistColumn[train_mask], trackColumn[train_mask])))
        URM_train = URM_train.tocsr()
        test_mask = np.logical_not(train_mask)
        URM_test = sps.coo_matrix((interactionColumn[test_mask], (playlistColumn[test_mask], trackColumn[test_mask])))
        URM_test = URM_test.tocsr()

        self.train_data['track_count'] = self.train_data.groupby('playlist_id')['playlist_id'].transform(
            pd.Series.value_counts)
        dfPlaylist = pd.DataFrame(data=self.train_data.loc[:, ['playlist_id', 'track_count']])
        dfPlaylist = dfPlaylist.sort_values('track_count', ascending=False)
        self.playlists = dfPlaylist.drop_duplicates(subset='playlist_id').reset_index().rename(
            columns={'index': 'interaction_id'})
        return URM_train, URM_test

    def precision(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
        return precision_score

    def recall(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
        return recall_score

    def meanAveragePrecision(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
        return map_score

    def evaluate(self, at=10):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0
        self.playlists = self.playlists['playlist_id'].values.reshape(len(self.playlists['playlist_id']), 1)
        for playlist_id in self.playlists:

            relevant_items = self.urm_test[playlist_id].indices

            if len(relevant_items) > 0:
                recommended_items = self.model.recommend(playlist_id, at=at)
                num_eval += 1

                cumulative_precision += self.precision(recommended_items, relevant_items)
                cumulative_recall += self.recall(recommended_items, relevant_items)
                cumulative_MAP += self.meanAveragePrecision(recommended_items, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))

    def import_data(self):
        pass

    def pipeline(self):
        self.model.fit()
        self.evaluate()


def run():
    pass


if __name__ == "__run__":
    run()


import numpy as np


class TopPopRecommender(object):

    def __init__(self,URM_train):
        self.urm_train = URM_train
        self.popularItems = None

    def fit(self):
        item_popularity = (self.urm_train > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(item_popularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, playlist_id, at=10):
        recommended_items = self.popularItems[0:at]
        return str(recommended_items).strip("[]")

import numpy as np

from base.RecommenderUtils import check_matrix
from base.BaseRecommender import RecommenderSystem


class TopPopRecommender(object):

    def __init__(self, URM_train):
        super(RecommenderSystem, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr')


    def fit(self):
        print("Training is started")
        item_popularity = (self.URM_train > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(item_popularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    # def recommend(self,playlist_id, at=10,exclude_seen=True):
    #     recommended_items = self.popularItems[0:at]
    #     return str(recommended_items).strip("[]")

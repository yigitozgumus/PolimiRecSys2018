import numpy as np
from base.BaseRecommender import RecommenderSystem
from base.RecommenderUtils import check_matrix
class TopPopRecommenderUnseen(RecommenderSystem):

    def __init__(self,URM_train):
        self.urm_train = URM_train
        self.popularItems = None

    def fit(self):

        itemPopularity = (self.urm_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()
        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self,playlist_id,at=10,remove_seen=True):
        if remove_seen:
            self.urm_train = check_matrix(self.urm_train,format="csr")
            unseen_items_mask = np.in1d(self.popularItems, self.urm_train[playlist_id].indices,
                                        assume_unique=True, invert=True)
            unseen_items = self.popularItems[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popularItems[0:at]

        return str(recommended_items).strip("[]")
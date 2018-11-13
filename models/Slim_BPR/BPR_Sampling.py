import numpy as np


class BPR_Sampling(object):

    def __init__(self):
        super(BPR_Sampling, self).__init__()

    def sample_user(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """
        while True:
            playlist_id = np.random.randint(0, self.num_users)
            numSeenItems = self.URM_train[playlist_id].nnz
            if 0 < numSeenItems < self.num_items:
                return playlist_id

    def sample_item_pair(self, playlist_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param playlist_id:
        :return: pos_item_id, neg_item_id
        """
        userSeenItems = self.URM_train[playlist_id].indices
        pos_item_id = userSeenItems[np.random.randint(0, len(userSeenItems))]
        while (True):
            neg_item_id = np.random.randint(0, self.num_items)
            if neg_item_id not in userSeenItems:
                return pos_item_id, neg_item_id

    def sample_triple(self):
        """
        Randomly samples a user and then samples randomly a seen and not seen item
        :return: user_id, pos_item_id, neg_item_id
        """
        user_id = self.sample_user()
        pos_item_id, neg_item_id = self.sample_item_pair(user_id)
        return user_id, pos_item_id, neg_item_id

    def initialize_fast_sampling(self, positive_threshold=1):
        print("Fast Sampling initialized")
        self.eligibleUsers = []
        self.userSeenItems = dict()
        # Select only positive interactions
        URM_train_positive = self.URM_train.multiply(self.URM_train >= positive_threshold)
        for playlist_id in range(self.num_users):
            if URM_train_positive[playlist_id].nnz > 0:
                self.eligibleUsers = np.append(self.eligibleUsers,playlist_id)
                self.userSeenItems[playlist_id] = URM_train_positive[playlist_id].indices
            self.eligibleUsers = np.array(self.eligibleUsers)


    def sampleBatch(self):
        playlist_id_list = np.random.choice(self.eligibleUsers, size=(self.batch_size))
        pos_item_id_list = [None] * self.batch_size
        neg_item_id_list = [None] * self.batch_size
        for sample_index in range(self.batch_size):
            user_id = playlist_id_list[sample_index]
            pos_item_id_list[sample_index] = np.random.choice(self.userSeenItems[user_id])
            negItemSelected = False
            # It's faster to just try again then to build a mapping of the non-seen items
            # for every user
            while (not negItemSelected):
                neg_item_id = np.random.randint(0, self.num_items)
                if (neg_item_id not in self.userSeenItems[user_id]):
                    negItemSelected = True
                    neg_item_id_list[sample_index] = neg_item_id

        return playlist_id_list, pos_item_id_list, neg_item_id_list

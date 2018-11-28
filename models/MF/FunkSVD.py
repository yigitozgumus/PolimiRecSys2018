import numpy as np
from base.RecommenderUtils import check_matrix

from base.BaseRecommender import RecommenderSystem
import models.MF.Cython.MF_RMSE as mf


class FunkSVD(RecommenderSystem):
    '''
    FunkSVD model
    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.
    '''

    # TODO: add global effects
    def __init__(self, URM_train):
        super(FunkSVD, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr', dtype=np.float32)
        self.parameters = None

    def __str__(self):
        return "FunkSVD Implementation"

    def fit(self, num_factors=50,
            learning_rate=0.01,
            reg=0.015,
            epochs=10,
            init_mean=0.0,
            init_std=0.1,
            lrate_decay=1.0,
            rnd_seed=42):
        """

        Initialize the model
        :param num_factors: number of latent factors
        :param learning_rate: initial learning rate used in SGD
        :param reg: regularization term
        :param epochs: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        """

        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

        self.U, self.V = mf.FunkSVD_sgd(self.URM_train, self.num_factors, self.learning_rate, self.reg, self.epochs,
                                        self.init_mean,
                                        self.init_std,
                                        self.lrate_decay, self.rnd_seed)
        self.parameters = "num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
                          "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.learning_rate, self.reg, self.epochs, self.init_mean, self.init_std,
            self.lrate_decay,
            self.rnd_seed)

    def recommendBatch(self, users_in_batch, n=None, exclude_seen=True):

        # compute the scores using the dot product
        user_profile_batch = self.URM_train[users_in_batch]
        scores_array = np.dot(self.U[users_in_batch], self.V.T)
        if self.normalize:
            raise ValueError("Not implemented")
        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if exclude_seen:
            scores_array[user_profile_batch.nonzero()] = -np.inf
        # if filterTopPop:
        #     scores_array[:,self.filterTopPop_ItemsID] = -np.inf
        # if filterCustomItems:
        #     scores_array[:, self.filterCustomItems_ItemsID] = -np.inf

        ranking = np.zeros((scores_array.shape[0], n), dtype=np.int)

        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]
            relevant_items_partition = (-scores).argpartition(n)[0:n]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]
        return ranking

    def recommend(self, user_id, n=None, exclude_seen=True):

        if n == None:
            n = self.URM_train.shape[1] - 1
        scores_array = np.dot(self.U[user_id], self.V.T)
        if self.normalize:
            raise ValueError("Not implemented")
        if exclude_seen:
            scores = self.filter_seen_on_scores(user_id, scores_array)

        # if filterTopPop:
        #     scores = self._filter_TopPop_on_scores(scores_array)
        #
        # if filterCustomItems:
        #     scores = self._filterCustomItems_on_scores(scores_array)

        relevant_items_partition = (-scores_array).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        return ranking

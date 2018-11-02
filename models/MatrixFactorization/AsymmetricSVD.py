import numpy as np
from base.RecommenderUtils import check_matrix

from base.BaseRecommender import RecommenderSystem
import models.MatrixFactorization.Cython.MF_RMSE as mf

class AsySVD(RecommenderSystem):
    '''
    AsymmetricSVD model
    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)

    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    '''

    # TODO: add global effects
    # TODO: recommendation for new-users. Update the precomputed profiles online
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 reg=0.015,
                 iters=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):

        super(AsySVD, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.reg = reg
        self.iters = iters
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed
        self.parameters = "num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={}".format(
            self.num_factors, self.lrate, self.reg, self.iters, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )

    def __str__(self):
        return "Asymmetric SVD Implementation"

    def fit(self, R):
        self.dataset = R
        R = check_matrix(R, 'csr', dtype=np.float32)
        self.X, self.Y = mf.AsySVD_sgd(R, self.num_factors, self.lrate, self.reg, self.iters, self.init_mean,
                                    self.init_std,
                                    self.lrate_decay, self.rnd_seed)
        # precompute the user factors
        M = R.shape[0]
        self.U = np.vstack([mf.AsySVD_compute_user_factors(R[i], self.Y) for i in range(M)])

    def recommend(self, playlist_id, n=None, exclude_seen=True,export =False):
        scores = np.dot(self.X, self.U[playlist_id].T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(playlist_id, ranking)
        if not export:
            return ranking[:n]
        else:
            return str(ranking[:n]).strip("[]")


    def _get_user_ratings(self, playlist_id):
        self.dataset = check_matrix(self.dataset,"csr")
        return self.dataset[playlist_id]

    def _get_item_ratings(self, track_id):
        self.dataset= check_matrix(self.dataset,"csc")
        return self.dataset[:, track_id]


    def _filter_seen(self, user_id, ranking):
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

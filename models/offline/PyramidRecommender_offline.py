"""
Author : Semsi Yigit Ozgumus
"""

# Libraries
import numpy as np
import pickle
from utils.OfflineDataLoader import OfflineDataLoader
# Models
from base.BaseRecommender import RecommenderSystem
from base.RecommenderUtils import check_matrix

from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender
from models.Slim_mark2.Cython.Slim_BPR_Cython import Slim_BPR_Recommender_Cython as Slim_mark2
from models.Slim_ElasticNet.SlimElasticNetRecommender import SLIMElasticNetRecommender
from models.graph.P3AlphaRecommender import P3alphaRecommender
from models.graph.RP3BetaRecommender import RP3betaRecommender


class PyramidRecommender_offline(RecommenderSystem):
    RECOMMENDER_NAME = "PyramidTreeRecommender_offline"

    def __init__(self, URM_train):
        super(PyramidRecommender_offline, self).__init__()
        self.URM_train = check_matrix(URM_train, "csr", dtype=np.float32)
        self.parameters = None
        self.dataset = None
        self.normalize = False

    def __repr__(self):
        return "Pyramid 3 Level Hybrid Offline Recommender"

    def fit(self,
            alpha=0.80849266253816,
            beta=0.7286503831547066,
            gamma=0.02895704968752022,
            chi = 1.8070865821028037,
            psi=4.256005405227253,
            omega=5.096018341419944,
            coeff = 39.966898886531645,
            normalize=False,
            save_model=False,
            submission=False,
            best_parameters=False):
        if best_parameters:
            m = OfflineDataLoader()
            folder_path, file_name = m.get_parameter(self.RECOMMENDER_NAME)
            self.loadModel(folder_path=folder_path, file_name=file_name)
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.chi = chi
            self.psi = psi
            self.omega = omega
            self.coeff = coeff

        self.normalize = normalize
        self.submission = not submission
        m = OfflineDataLoader()
        self.m_user_knn_cf = UserKNNCFRecommender(self.URM_train)
        folder_path_ucf, file_name_ucf = m.get_model(UserKNNCFRecommender.RECOMMENDER_NAME, training=self.submission)
        self.m_user_knn_cf.loadModel(folder_path=folder_path_ucf, file_name=file_name_ucf)

        self.m_item_knn_cf = ItemKNNCFRecommender(self.URM_train)
        folder_path_icf, file_name_icf = m.get_model(ItemKNNCFRecommender.RECOMMENDER_NAME, training=self.submission)
        self.m_item_knn_cf.loadModel(folder_path=folder_path_icf, file_name=file_name_icf)

        self.m_slim_mark2 = Slim_mark2(self.URM_train)
        folder_path_slim, file_name_slim = m.get_model(Slim_mark2.RECOMMENDER_NAME, training=self.submission)
        self.m_slim_mark2.loadModel(folder_path=folder_path_slim, file_name=file_name_slim)

        self.m_alpha = P3alphaRecommender(self.URM_train)
        folder_path_alpha, file_name_alpha = m.get_model(P3alphaRecommender.RECOMMENDER_NAME, training=self.submission)
        self.m_alpha.loadModel(folder_path=folder_path_alpha, file_name=file_name_alpha)

        self.m_beta = RP3betaRecommender(self.URM_train)
        folder_path_beta, file_name_beta = m.get_model(RP3betaRecommender.RECOMMENDER_NAME, training=self.submission)
        self.m_beta.loadModel(folder_path=folder_path_beta, file_name=file_name_beta)

        self.m_slim_elastic = SLIMElasticNetRecommender(self.URM_train)
        folder_path_elastic, file_name_elastic = m.get_model(SLIMElasticNetRecommender.RECOMMENDER_NAME,
                                                             training=self.submission)
        self.m_slim_elastic.loadModel(folder_path=folder_path_elastic, file_name=file_name_elastic)

        self.W_sparse_URM = check_matrix(self.m_user_knn_cf.W_sparse, "csr", dtype=np.float32)
        self.W_sparse_URM_T = check_matrix(self.m_item_knn_cf.W_sparse, "csr", dtype=np.float32)
        self.W_sparse_Slim = check_matrix(self.m_slim_mark2.W_sparse, "csr", dtype=np.float32)
        self.W_sparse_alpha = check_matrix(self.m_alpha.W_sparse, "csr", dtype=np.float32)
        self.W_sparse_beta = check_matrix(self.m_beta.W_sparse, "csr", dtype=np.float32)
        self.W_sparse_elastic = check_matrix(self.m_slim_elastic.W_sparse, "csr", dtype=np.float32)
        # Precomputations
        self.matrix_alpha_beta = self.alpha * self.W_sparse_alpha + (3 - self.alpha) * self.W_sparse_beta
        self.matrix_slim = self.beta * self.W_sparse_Slim + ((3 - self.beta) * self.W_sparse_elastic * self.coeff)

        self.parameters = "alpha={}, beta={}, gamma={}, chi={}, psi={}, omega={}, coeff={}".format(self.alpha, self.beta, self.gamma,
                                                                                   self.chi, self.psi, self.omega, self.coeff)
        if save_model:
            self.saveModel("saved_models/submission/", file_name=self.RECOMMENDER_NAME)

    def recommend(self, playlist_id_array, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False,
                  remove_CustomItems_flag=False, export=False):
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(playlist_id_array):
            playlist_id_array = np.atleast_1d(playlist_id_array)
            single_user = True
        else:
            single_user = False
        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        scores_users = self.W_sparse_URM[playlist_id_array].dot(self.URM_train).toarray()
        scores_items = self.URM_train[playlist_id_array].dot(self.W_sparse_URM_T).toarray()
        scores_knn = self.gamma * scores_users + (3-self.gamma) * scores_items
        scores_ab = self.URM_train[playlist_id_array].dot(self.matrix_alpha_beta).toarray()
        scores_slim = self.URM_train[playlist_id_array].dot(self.matrix_slim).toarray()
        scores = self.chi * scores_knn + self.psi * scores_ab + self.omega * scores_slim

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            user_profile = self.URM_train[playlist_id_array]
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                # print(rated.shape)
                # print(self.W_sparse.shape)
                den = rated.dot(self.W_sparse).toarray()
            else:
                den = rated.dot(self.W)
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den

        for user_index in range(len(playlist_id_array)):
            user_id = playlist_id_array[user_index]
            if remove_seen_flag:
                scores[user_index, :] = self._remove_seen_on_scores(user_id, scores[user_index, :])

        relevant_items_partition = (-scores).argpartition(cutoff, axis=1)[:, 0:cutoff]
        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores[
            np.arange(scores.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            if not export:
                return ranking_list
            elif export:
                return str(ranking_list[0]).strip("[,]")

        if not export:
            return ranking_list
        elif export:
            return str(ranking_list).strip("[,]")

    def saveModel(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))
        dictionary_to_save = {"W_sparse_URM": self.W_sparse_URM,
                              "W_sparse_URM_T": self.W_sparse_URM_T,
                              "W_sparse_Slim": self.W_sparse_Slim,
                              "W_sparse_alpha": self.W_sparse_alpha,
                              "W_sparse_beta": self.W_sparse_beta,
                              "W_sparse_elastic": self.W_sparse_elastic,
                              "matrix_slim": self.matrix_slim,
                              "matrix_alpha_beta": self.matrix_alpha_beta,
                              "alpha": self.alpha,
                              "beta": self.beta,
                              "gamma": self.gamma,
                              "chi": self.chi,
                              "psi": self.psi,
                              "omega": self.omega}

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))

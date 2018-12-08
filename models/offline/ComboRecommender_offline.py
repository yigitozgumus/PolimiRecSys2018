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

from models.offline.PartyRecommender_offline import PartyRecommender_offline #0.09217
from models.offline.PyramidRecommender_offline import PyramidRecommender_offline #0.09285
from models.offline.PyramidItemTreeRecommender_offline import PyramidItemTreeRecommender_offline #0.09234
from models.offline.HybridEightRecommender_offline import HybridEightRecommender_offline #0.09298
from models.offline.SingleNeuronRecommender_offline import SingleNeuronRecommender_offline #0.09373



class ComboRecommender_offline(RecommenderSystem):
    RECOMMENDER_NAME = "ComboRecommender_offline"

    def __init__(self, URM_train,ICM):
        super(ComboRecommender_offline, self).__init__()
        self.URM_train = check_matrix(URM_train, "csr", dtype=np.float32)
        self.ICM = check_matrix(ICM,"csr",dtype=np.float32)
        self.parameters = None
        self.dataset = None
        self.normalize = False

    def __repr__(self):
        return "Combo Recommender"

    def fit(self,
            alpha=0.1,
            beta=0.1,
            gamma=0.1,
            theta=0.1,
            delta=0.1,
            normalize=False,
            save_model=False,
            submission=False,
            best_parameters=False,
            location="submission"):
        if best_parameters:
            m = OfflineDataLoader()
            folder_path, file_name = m.get_parameter(self.RECOMMENDER_NAME)
            self.loadModel(folder_path=folder_path, file_name=file_name)
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.theta = theta
            self.delta= delta

        self.normalize = normalize
        self.submission = not submission
        m = OfflineDataLoader()
        self.m_party = PartyRecommender_offline(self.URM_train)
        folder_path_ucf, file_name_ucf = m.get_model(PartyRecommender_offline.RECOMMENDER_NAME, training=self.submission)
        self.m_party.loadModel(folder_path=folder_path_ucf, file_name=file_name_ucf)

        self.m_pyramid = PyramidRecommender_offline(self.URM_train)
        folder_path_icf, file_name_icf = m.get_model(PyramidRecommender_offline.RECOMMENDER_NAME, training=self.submission)
        self.m_pyramid.loadModel(folder_path=folder_path_icf, file_name=file_name_icf)

        self.m_pyitem = PyramidItemTreeRecommender_offline(self.URM_train,self.ICM)
        folder_path_slim, file_name_slim = m.get_model(PyramidItemTreeRecommender_offline.RECOMMENDER_NAME, training=self.submission)
        self.m_pyitem.loadModel(folder_path=folder_path_slim, file_name=file_name_slim)

        self.m_8 = HybridEightRecommender_offline(self.URM_train,self.ICM)
        folder_path_alpha, file_name_alpha = m.get_model(HybridEightRecommender_offline.RECOMMENDER_NAME, training=self.submission)
        self.m_8.loadModel(folder_path=folder_path_alpha, file_name=file_name_alpha)

        self.m_sn = SingleNeuronRecommender_offline(self.URM_train,self.ICM)
        folder_path_alpha, file_name_alpha = m.get_model(SingleNeuronRecommender_offline.RECOMMENDER_NAME, training=self.submission)
        self.m_sn.loadModel(folder_path=folder_path_alpha, file_name=file_name_alpha)

        self.parameters = "alpha={}, beta={}, gamma={}, theta={},delta={} ".format(self.alpha, self.beta, self.gamma,
                                                                                   self.theta,self.delta)
        if save_model:
            self.saveModel("saved_models/"+location+"/", file_name=self.RECOMMENDER_NAME)

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

        # Party Score Calculation
        scores_URM = self.m_party.W_sparse_URM[playlist_id_array].dot(self.URM_train).toarray()
        scores_alphabeta = self.URM_train[playlist_id_array].dot(self.m_party.matrix_alpha_beta).toarray()
        scores_level1 = self.URM_train[playlist_id_array].dot(self.m_party.matrix_level1).toarray()
        scores_level2 = self.m_party.gamma * scores_alphabeta + (1 - self.m_party.gamma) * scores_URM
        scores_level3 = self.m_party.theta * scores_level2 + (1 - self.m_party.theta) * scores_level1
        scores_elastic = self.URM_train[playlist_id_array].dot(self.m_party.W_sparse_elastic).toarray()
        scores_party = (self.m_party.omega * self.m_party.coeff * scores_elastic) + (1 - self.m_party.omega) * scores_level3
        # Pyramid Score Calculation
        scores_users = self.m_pyramid.W_sparse_URM[playlist_id_array].dot(self.URM_train).toarray()
        scores_items = self.URM_train[playlist_id_array].dot(self.m_pyramid.W_sparse_URM_T).toarray()
        scores_knn = self.m_pyramid.gamma * scores_users + (1 - self.m_pyramid.gamma) * scores_items
        scores_ab = self.URM_train[playlist_id_array].dot(self.m_pyramid.matrix_alpha_beta).toarray()
        scores_slim = self.URM_train[playlist_id_array].dot(self.m_pyramid.matrix_slim).toarray()
        scores_pyramid = self.m_pyramid.chi * scores_knn + self.m_pyramid.psi * scores_ab + self.m_pyramid.omega * scores_slim
        # Pyramid Item Tree Score Calculation
        scores_users = self.m_pyitem.W_sparse_URM[playlist_id_array].dot(self.URM_train).toarray()
        scores_items_cf = self.URM_train[playlist_id_array].dot(self.m_pyitem.W_sparse_URM_T).toarray()
        scores_items_cbf = self.URM_train[playlist_id_array].dot(self.m_pyitem.W_sparse_ICM).toarray()
        scores_knn = self.m_pyitem.gamma * scores_users + (1 - self.m_pyitem.gamma) * scores_items_cf + self.m_pyitem.tau * scores_items_cbf
        scores_ab = self.URM_train[playlist_id_array].dot(self.m_pyitem.matrix_alpha_beta).toarray()
        scores_slim = self.URM_train[playlist_id_array].dot(self.m_pyitem.matrix_slim).toarray()
        scores_pyitem = self.m_pyitem.chi * scores_knn + self.m_pyitem.psi * scores_ab + self.m_pyitem.omega * scores_slim
        # Hybrid Eight Score Calculation
        scores_users = self.m_8.W_sparse_URM[playlist_id_array].dot(self.URM_train).toarray()
        scores_wo_users = self.URM_train[playlist_id_array].dot(self.m_8.matrix_wo_user).toarray()
        scores_knn = self.m_8.gamma * scores_users + scores_wo_users
        scores_ab = self.URM_train[playlist_id_array].dot(self.m_8.matrix_alpha_beta).toarray()
        scores_slim = self.URM_train[playlist_id_array].dot(self.m_8.matrix_slim).toarray()
        scores_8 = self.m_8.chi * scores_ab + self.m_8.psi * scores_knn + self.m_8.omega * scores_slim
        # Single Neuron Recommender Score Calculation
        scores_users = self.m_sn.W_sparse_URM[playlist_id_array].dot(self.URM_train).toarray()
        scores_wo_user = self.URM_train[playlist_id_array].dot(self.m_sn.matrix_wo_user).toarray()
        scores_sn = scores_users + scores_wo_user
        # Total prediction
        scores = self.alpha * scores_party + self.beta * scores_pyramid + self.gamma * scores_pyitem + self.theta * scores_8 + self.delta * scores_sn
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
        dictionary_to_save = {
                              "m_party": self.m_party,
                              "m_pyramid" : self.m_pyramid,
                              "m_pyitem" : self.m_pyitem,
                              "m_8": self.m_8,
                              "m_sn": self.m_sn,
                              "alpha": self.alpha,
                              "beta": self.beta,
                              "gamma": self.gamma,
                              "theta": self.theta,
                              "delta":self.delta
                              }

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))

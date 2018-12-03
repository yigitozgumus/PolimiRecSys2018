
import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize

from utils.OfflineDataLoader import OfflineDataLoader
from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix, similarityMatrixTopK


import time, sys

class P3alphaRecommender(RecommenderSystem,RecommenderSystem_SM):
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3_Alpha_Recommender"
    def __init__(self, URM_train):
        super(P3alphaRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, format='csr', dtype=np.float32)
        self.sparse_weights = True
        self.parameters = None
        self.dataset=None


    def __repr__(self):
        return "P3alpha Recommender"

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False,save_model=False,best_parameters=False):
        if best_parameters:
            m = OfflineDataLoader()
            folder_alpha, file_alpha = m.get_parameter(self.RECOMMENDER_NAME)
            self.loadModel(folder_path=folder_alpha,file_name=file_alpha)
        else:
            self.topK = topK
            self.alpha = alpha
            self.normalize_similarity = normalize_similarity

        self.min_rating = min_rating
        self.implicit = implicit
        self.parameters = "alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                            self.min_rating, self.topK, self.implicit,
                                                                            self.normalize_similarity)

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1


            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.coo_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))
        self.W_sparse = check_matrix(self.W_sparse,"csr",dtype=np.float32)


        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, forceSparseOutput = True, k=self.topK)
            self.sparse_weights = True

        if save_model:
            self.saveModel("saved_models/submission/",file_name="P3_Alpha_Recommender_submission_model")

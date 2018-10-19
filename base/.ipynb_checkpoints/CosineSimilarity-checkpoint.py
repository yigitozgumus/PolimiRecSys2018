import numpy as np
import time
import scipy.sparse as sps
import sys
from sklearn.metrics.pairwise import cosine_similarity

from base.RecommenderUtils import check_matrix


class CosineSimilarity(object):

    def __init__(self,dataMatrix, neighbourhood=100, shrink = True, normalize = True, mode = "cosine",batchSize = 100):
        """

        :param dataMatrix:
        :param neighbourhood:
        :param shrink:
        :param normalize:
        :param mode: This can be either cosine, pearson or adjusted (cosine)
        :param batchSize:
        """
        super(CosineSimilarity, self).__init__()
        self.neighbourhood = neighbourhood
        self.shrink = shrink
        self.normalize = normalize
        self.mode = mode
        self.batchSize = batchSize

        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]

        self.dataMatrix = dataMatrix.copy()

        if mode == "adjusted":
            self.adjustedCosine = True
        elif mode == "pearson":
            self.pearsonCorrelation = True
        elif mode == "cosine":
            pass

        if self.neighbourhood == 0:
            self.simFull = np.zeros((self.n_columns,self.n_columns))


    def computeCosineSimilarity(self):
        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        sumOfSquared = np.sqrt(sumOfSquared)

        for columnIndex in range(self.n_columns):
            processedItems +=1

            #Timing show for the terminal
            if time.time() - start_time_print_batch >= 30 or processedItems == self.n_columns:
                columnPerSec = processedItems / (time.time() - start_time)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / self.n_columns * 100, columnPerSec,
                                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            # All data points for a given item
            item_data = self.dataMatrix[:, columnIndex]
            item_data = item_data.toarray().squeeze()
            # Compute item similarities
            this_column_weights = self.dataMatrix.T.dot(item_data)
            this_column_weights[columnIndex] = 0.0
            if self.normalize:
                denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)
                # If no normalization or tanimoto is selected, apply only shrink
            elif self.shrink != 0:
                this_column_weights = this_column_weights / self.shrink

            if self.neighbourhood == 0:
                self.simFull[:, columnIndex] = this_column_weights

            else:
                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.neighbourhood - 1)[0:self.neighbourhood]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix
                values.extend(this_column_weights[top_k_idx])
                rows.extend(top_k_idx)
                cols.extend(np.ones(self.neighbourhood) * columnIndex)

        if self.neighbourhood== 0:
            return self.simFull

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_columns, self.n_columns),
                                      dtype=np.float32)

            return W_sparse

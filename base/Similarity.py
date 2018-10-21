import numpy as np
import time
import scipy.sparse as sps
import sys
from sklearn.metrics.pairwise import cosine_similarity

from base.RecommenderUtils import check_matrix


class Similarity(object):

    def __init__(self,dataMatrix, neighbourhood=100, shrink = 0, mode = None,batchSize = 100,verbose=False):
        """

        :param dataMatrix:
        :param neighbourhood:
        :param shrink:
        :param normalize:
        :param mode: This can be either cosine, pearson or adjusted (cosine)
        :param batchSize:
        """
        super(Similarity, self).__init__()
        self.neighbourhood = neighbourhood
        self.shrink = shrink
        self.verbose = verbose
        self.batchSize = batchSize
        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.adjustedCosine = False
        self.pearsonCorrelation = False
        self.method = mode


        self.dataMatrix = dataMatrix.copy()

        if mode == "adjusted":
            self.adjustedCosine = True
        elif mode == "pearson":
            self.pearsonCorrelation = True
        elif mode == "cosine":
            pass

        if self.neighbourhood == 0:
            self.full_weights = np.zeros((self.n_rows,self.n_rows))

    def normalizeData_meanReduce(self,blockSize=1000,mode="user"):
        if mode == "normal":
            return
        if self.verbose:
            print("Normalization of data started with {} mean rating".format(mode))
        start_time = time.time()
        mode_ = 1 if mode == "user" else 0
        if not mode_:
            self.dataMatrix = check_matrix(self.dataMatrix, 'csc')
        else:
            self.dataMatrix = check_matrix(self.dataMatrix, 'csr')
        interactionsPerVector = np.diff(self.dataMatrix.indptr)
        sumPerVector = np.asarray(self.dataMatrix.sum(axis=mode_)).ravel()
        sumPerVector =  np.sqrt(sumPerVector)
        nonZeroVectors = interactionsPerVector > 0

        vectorAverage = np.zeros_like(sumPerVector)
        vectorAverage[nonZeroVectors] = sumPerVector[nonZeroVectors]
        start_ = 0
        end_   = 0
        while end_ < self.dataMatrix.shape[1 - mode_]:
            end_ = min(self.dataMatrix.shape[1-mode_], end_ + blockSize)
        #    self.dataMatrix.data[self.dataMatrix.indptr[start_]:self.dataMatrix.indptr[end_]] -= 1
            self.dataMatrix.data[self.dataMatrix.indptr[start_]:self.dataMatrix.indptr[end_]] -= \
          (1/  np.repeat(vectorAverage[start_:end_],interactionsPerVector[start_:end_]))
            start_ += blockSize
        if self.verbose:
            print("Normalization of data is completed in {} seconds".format(time.time() - start_time))

    def computeUUSimilarity(self):
        # define the output csr matrix values
        values = []
        rows = []
        cols = []
        if self.verbose:
            print("Computation of User User Similarity matrix with {} mode is started.".format((self.method)))
        # Timer initializations
        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.pearsonCorrelation:
            self.normalizeData_meanReduce()
        if self.adjustedCosine:
            self.normalizeData_meanReduce(mode="item")
        
        # self.dataMatrix = check_matrix(self.dataMatrix, 'csr')
        # sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
        # sumOfSquared = np.sqrt(sumOfSquared)

        for rowIndex in range(self.n_rows):
            processedItems +=1

            #Timing show for the terminal
            if time.time() - start_time_print_batch >= 30 or processedItems == self.n_rows:
                rowPerSec = processedItems / (time.time() - start_time)

                print("Similarity row {} ( {:2.0f} % ), {:.2f} row/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / self.n_rows * 100, rowPerSec,
                                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            # All data points for a given item
            item_data = self.dataMatrix[rowIndex,:]
            item_data = item_data.toarray().squeeze()
            # Compute item similarities
            this_row_weights = self.dataMatrix.dot(item_data)
            this_row_weights[rowIndex] = 0.0
            if self.shrink != 0:
                this_row_weights = this_row_weights / self.shrink

            if self.neighbourhood == 0:
                self.full_weights[rowIndex,:] = this_row_weights

            else:
                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_row_weights).argpartition(self.neighbourhood - 1)[0:self.neighbourhood]
                relevant_items_partition_sorting = np.argsort(-this_row_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix
                values.extend(this_row_weights[top_k_idx])
                rows.extend(np.ones(self.neighbourhood) * rowIndex)
                cols.extend(top_k_idx)
        if self.verbose:
            print("Computation is completend in {} minutes".format((time.time() - start_time)/60))

        if self.neighbourhood== 0:
            return self.full_weights

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_rows, self.n_rows),
                                      dtype=np.float32)

            return W_sparse

    def computeIISimilarity(self):
        # define the output csr matrix values
        values = []
        rows = []
        cols = []
        if self.verbose:
            print("Computation of Item Item Similarity matrix with {} mode is started.".format((self.method)))
        # Timer initializations
        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.adjustedCosine:
            self.normalizeData_meanReduce(mode='item')

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(sumOfSquared)

        for columnIndex in range(self.n_columns):
            processedItems += 1

            #Timing show for the terminal
            if time.time() - start_time_print_batch >= 30 or processedItems == self.n_columns:
                columnPerSec = processedItems / (time.time() - start_time)

                print("Similarity row {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / self.n_columns * 100, columnPerSec,
                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            # All data points for a given item
            item_data = self.dataMatrix[:,columnIndex]
            item_data = item_data.toarray().squeeze()
            # Compute item similarities
            this_column_weights = self.dataMatrix.T.dot(item_data)
            this_column_weights[columnIndex] = 0.0
            
            if self.shrink != 0:
                this_column_weights = this_column_weights / self.shrink

            if self.neighbourhood == 0:
                self.full_weights[:,columnIndex] = this_column_weights

            else:
                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(
                    self.neighbourhood - 1)[0:self.neighbourhood]
                relevant_items_partition_sorting = np.argsort(
                    -this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix
                values.extend(this_column_weights[top_k_idx])
                rows.extend(np.ones(self.neighbourhood) * columnIndex)
                cols.extend(top_k_idx)
        if self.verbose:
            print("Computation is completend in {} minutes".format((time.time() - start_time)/60))


        if self.neighbourhood == 0:
            return self.full_weights

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_columns, self.n_columns),
                                      dtype=np.float32)

            return W_sparse

import numpy as np
import scipy.sparse as sps
import time
import os
from sklearn import feature_extraction
from sklearn.preprocessing import normalize


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


def extract_UCM(URM_train):
    URM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM_train.T)
    URM_tfidf = URM_tfidf.T
    return URM_tfidf

def similarityMatrixTopK(item_weights, forceSparseOutput=True, k=100, verbose=False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"
    start_time = time.time()
    if verbose:
        print("Generating topK matrix")
    nitems = item_weights.shape[1]
    k = min(k, nitems)
    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)
    if not sparse_weights:
        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column
        if inplace:
            W = item_weights
        else:
            W = item_weights.copy()
        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0
        if forceSparseOutput:
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))
            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))
            return W_sparse
        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))
        return W
    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []
        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)
        for item_idx in range(nitems):
            cols_indptr.append(len(data))
            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx + 1]
            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]
            idx_sorted = np.argsort(column_data)  # sort by column
            top_k_idx = idx_sorted[-k:]
            data.extend(column_data[top_k_idx])
            rows_indices.extend(column_row_index[top_k_idx])
        cols_indptr.append(len(data))
        # During testing CSR is faster
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()
        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))
        return W_sparse

def removeTopPop(URM_1, URM_2=None, percentageToRemove=0.2):
    """
    Remove the top popular items from the matrix
    :param URM_1: user X items
    :param URM_2: user X items
    :param percentageToRemove: value 1 corresponds to 100%
    :return: URM: user X selectedItems, obtained from URM_1
             Array: itemMappings[selectedItemIndex] = originalItemIndex
             Array: removedItems
    """
    item_pop = URM_1.sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)

    if URM_2 != None:
        assert URM_2.shape[1] == URM_1.shape[1], \
            "The two URM do not contain the same number of columns, URM_1 has {}, URM_2 has {}".format(
                URM_1.shape[1], URM_2.shape[1])

        item_pop += URM_2.sum(axis=0)

    item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems,)
    popularItemsSorted = np.argsort(item_pop)[::-1]

    numItemsToRemove = int(len(popularItemsSorted) * percentageToRemove)

    # Choose which columns to keep
    itemMask = np.in1d(np.arange(len(popularItemsSorted)), popularItemsSorted[:numItemsToRemove], invert=True)

    # Map the column index of the new URM to the original ItemID
    itemMappings = np.arange(len(popularItemsSorted))[itemMask]

    removedItems = np.arange(len(popularItemsSorted))[np.logical_not(itemMask)]

    return URM_1[:, itemMask], itemMappings, removedItems

def to_tfidf(dataMatrix):
    dataMatrix = sps.coo_matrix(dataMatrix).T
    ICM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(dataMatrix)
    ICM_tfidf = ICM_tfidf_T.T
    ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')
    return ICM_tfidf.tocsr()

def to_okapi(dataMatrix, K1=1.2, B=0.75):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :param K1:
    :param B:
    :return:
    """

    assert B>0 and B<1, "okapi_BM_25: B must be in (0,1)"
    assert K1>0,        "okapi_BM_25: K1 must be > 0"


    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)

    dataMatrix = sps.coo_matrix(dataMatrix)

    N = float(dataMatrix.shape[0])
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # calculate length_norm per document
    row_sums = np.ravel(dataMatrix.sum(axis=1))

    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    dataMatrix.data = dataMatrix.data * (K1 + 1.0) / (K1 * length_norm[dataMatrix.row] + dataMatrix.data) * idf[dataMatrix.col]

    return dataMatrix.tocsr()


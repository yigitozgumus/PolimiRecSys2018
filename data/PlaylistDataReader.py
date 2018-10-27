import numpy as np
import scipy.sparse as sps
import pandas as pd


def loadCSVintoSparse(filePath, header=True, separator=","):
    values, rows, cols = [], [], []

    fileHandle = open(filePath, "r")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 100000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")
            rows.append(int(line[0]))
            cols.append(int(line[1]))
            values.append(float(1))

    fileHandle.close()

    return sps.csr_matrix((values, (rows, cols)), dtype=np.float32)


def saveSparseIntoCSV(filePath, sparse_matrix, separator=","):
    sparse_matrix = sparse_matrix.tocoo()

    fileHandle = open(filePath, "w")

    for index in range(len(sparse_matrix.data)):
        fileHandle.write("{row}{separator}{col}{separator}{value}\n".format(
            row=sparse_matrix.row[index], col=sparse_matrix.col[index], value=sparse_matrix.data[index],
            separator=separator))


class PlaylistDataReader(object):

    def __init__(self,
                 splitTrainTest=True,
                 trainTestSplit=0.8,
                 loadPredefinedTrainTest=True,
                 verbose=False,
                 adjustSequentials=False):
        super(PlaylistDataReader, self).__init__()
        self.verbose = verbose
        self.loadPredefined = loadPredefinedTrainTest
        self.URM_train = None
        self.URM_test = None
        self.URM_all = None

        if trainTestSplit >= 1.0:
            raise ValueError("PlaylistDaraReader: splitTrainTestSplit" \
                             " must be a value smaller than one.")

        if verbose:
            print("PlaylistDataReader: Data is loading. . .")

        dataSubfolder = "./data/"
        train_path = "./data/train.csv"
        track_path = "./data/tracks.csv"
        target_path = "./data/target_playlists.csv"

        try:
            self.trainData = pd.read_csv(dataSubfolder + "train.csv")
            self.trackData = pd.read_csv(dataSubfolder + "tracks.csv")
            self.targetData = pd.read_csv(dataSubfolder + "target_playlists.csv")

        except FileNotFoundError:
            print("PlaylistDataReader: train.csv or tracks.csv or target_playlists.csv not.")

        if self.loadPredefined:
            try:
                self.URM_train = sps.load_npz(dataSubfolder + "URM_train.npz")
                self.URM_test = sps.load_npz(dataSubfolder + "URM_test.npz")
                return
            except FileNotFoundError:
                print("PlaylistDataReader: URM_train or URM_test not.Found. Building new ones")
                if adjustSequentials:
                    self.trainData['popularity'] = 3
                    sequentials = self.targetData.loc[:4999, 'playlist_id']
                    # apply the ratings
                    for seq in sequentials:
                        length = len(self.trainData[self.trainData['playlist_id'] == seq])
                        self.trainData.loc[self.trainData['playlist_id'] == seq, 'popularity'] = np.linspace(5, 1,
                                                                                                             length)
                    interaction = np.array(self.trainData['popularity'])
                    rows = np.array(self.trainData['playlist_id'])
                    cols = np.array(self.trainData['track_id'])
                    self.URM_all = sps.csr_matrix((interaction, (rows, cols)))
                else:
                    self.trainData['popularity'] = 1
                    interaction = np.array(self.trainData['popularity'])
                    rows = np.array(self.trainData['playlist_id'])
                    cols = np.array(self.trainData['track_id'])
                    self.URM_all = sps.csr_matrix((interaction, (rows, cols)))

        if splitTrainTest:
            self.URM_all = self.URM_all.tocoo()
            numInteractions = len(self.URM_all.data)

            trainMask = np.random.choice([True, False], numInteractions, p=[trainTestSplit, 1 - trainTestSplit])
            self.URM_train = sps.coo_matrix(
                (self.URM_all.data[trainMask], (self.URM_all.row[trainMask], self.URM_all.col[trainMask])))
            testMask = np.logical_not(trainMask)
            self.URM_test = sps.coo_matrix(
                (self.URM_all.data[testMask], (self.URM_all.row[testMask], self.URM_all.col[testMask])))

            del self.URM_all
            print("PlaylistDataReader: saving URM_train and URM_test")
            #sps.save_npz(dataSubfolder + "URM_train.npz", self.URM_train)
            #sps.save_npz(dataSubfolder + "URM_test.npz", self.URM_test)
        if verbose:
            print("PlaylistDataReader: Data loading is complete")

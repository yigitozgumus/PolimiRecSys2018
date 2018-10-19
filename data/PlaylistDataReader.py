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

    def __init__(self, splitTrainTest=False,
                 splitTrainTestValidation=[0.8, 0.1, 0.1],
                 loadPredefinedTrainTest=True):
        super(PlaylistDataReader, self).__init__()

        if sum(splitTrainTestValidation) != 1.0 or len(splitTrainTestValidation) != 3:
            raise ValueError("PlaylistDaraReader: splitTrainTestValidation" \
                             " must be a probability distribution over Train, Test and Validation")

        print("PlaylistDataReader: Data is loading. . . \n")

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

        if not loadPredefinedTrainTest:
            pass
        else:
            try:
                self.URM_train = sps.load_npz(dataSubfolder + "URM_train.npz")
                self.URM_test = sps.load_npz(dataSubfolder + "URM_test.npz")
                self.URM_validation = sps.load_npz(dataSubfolder + "URM_validation.npz")
                return
            except FileNotFoundError:
                print("PlaylistDataReader: URM_train or URM_test or URM_validation not.Found. Building new ones\n")
                splitTrainTest = True
                # TODO
                self.URM_all = loadCSVintoSparse(train_path)

        if splitTrainTest:
            self.URM_all = self.URM_all.tocoo()
            numInteractions = len(self.URM_all.data)
            split = np.random.choice([1, 2, 3], numInteractions, p=splitTrainTestValidation)

            trainMask = (split == 1)
            interaction = self.URM_all.data[trainMask]
            row = self.URM_all.row[trainMask]
            col = self.URM_all.col[trainMask]
            self.URM_train = sps.coo_matrix((interaction, (row, col)))
            self.URM_train - self.URM_train.tocsr()

            testMask = (split == 2)
            interaction = self.URM_all.data[testMask]
            row = self.URM_all.row[testMask]
            col = self.URM_all.col[testMask]
            self.URM_test = sps.coo_matrix((interaction, (row, col)))
            self.URM_test - self.URM_test.tocsr()

            #
            validationMask = (split == 3)
            interaction = self.URM_all.data[validationMask]
            row = self.URM_all.row[validationMask]
            col = self.URM_all.col[validationMask]
            self.URM_validation = sps.coo_matrix((interaction, (row, col)))
            self.URM_validation - self.URM_validation.tocsr()


            del self.URM_all

            print("Movielens10MReader: saving URM_train and URM_test\n")
            sps.save_npz(dataSubfolder + "URM_train.npz", self.URM_train)
            sps.save_npz(dataSubfolder + "URM_test.npz", self.URM_test)
            sps.save_npz(dataSubfolder + "URM_validation.npz", self.URM_validation)
        print("PlaylistDataReader: Data loading is complete\n")

    def get_URM_train(self):
        return self.URM_train

    def get_URM_test(self):
        return self.URM_test

    def get_URM_validation(self):
        return self.URM_validation

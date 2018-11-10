import numpy as np
import scipy.sparse as sps
import pandas as pd
from sklearn import feature_extraction
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.utils import shuffle

class PlaylistDataReader(object):

    def __init__(self,
                 loadPredefinedTrainTest=True,
                 verbose=False,
                 adjustSequentials=False):
        super(PlaylistDataReader, self).__init__()
        self.verbose = verbose
        self.loadPredefined = loadPredefinedTrainTest
        self.URM_train = None
        self.URM_test = None
        self.URM_all = None
        self.ICM = None
        self.UCM = None
        self.adjust = adjustSequentials
        if verbose:
            print("PlaylistDataReader: Data is loading. . .")
        self.dataSubfolder = "./data/"
        self.trainData = pd.read_csv(self.dataSubfolder + "train.csv")
        self.trackData = pd.read_csv(self.dataSubfolder + "tracks.csv")
        self.targetData = pd.read_csv(self.dataSubfolder + "target_playlists.csv")
        return

        ## Attribute Methods

    def get_trackData(self):
        return self.trackData

    def get_tracks(self):
        tracks = self.trackData['track_id'].unique()
        return np.sort(tracks)

    def get_playlists(self):
        playlists = self.trainData['playlist_id'].unique()
        return np.sort(playlists)

    def get_target_playlists(self):
        target_playlists = self.targetData['playlist_id'].unique()
        return np.sort(target_playlists)

    def get_artists(self):
        artists = self.trackData['artist_id'].unique()
        return np.sort(artists)

    def get_albums(self):
        albums = self.trackData['album_id'].unique()
        return np.sort(albums)

    def get_URM_train(self):
        if self.URM_train is None:
            raise TypeError("URM_train is not built")
        else:
            return self.URM_train

    def get_URM_test(self):
        if self.URM_test is None:
            raise TypeError("URM_test is not built")
        else:
            return self.URM_test

    def get_ICM(self):
        if self.ICM is None:
            raise TypeError("ICM is not built")
        else:
            return self.ICM
    def get_UCM(self):
        if self.UCM is None:
            raise TypeError("UCM is not built")
        else:
            return self.UCM

    def build_URM(self):
        print("PlaylistDataReader: URM Matrix is being built...")
        # if self.loadPredefined:
        #     try:
        #         self.URM_train = sps.load_npz(self.dataSubfolder + "URM_train.npz")
        #         self.URM_test = sps.load_npz(self.dataSubfolder + "URM_test.npz")
        #         print("URM's are imported from the saved data.")
        #         return
        #     except FileNotFoundError:
        #         print("PlaylistDataReader: URM_train or URM_test not.Found. Building new ones")
        if self.adjust:
            self.trainData['popularity'] = 3
            sequentials = self.targetData.loc[:4999, 'playlist_id']
            # apply the ratings
            for seq in sequentials:
                length = len(self.trainData[self.trainData['playlist_id'] == seq])
                self.trainData.loc[self.trainData['playlist_id'] == seq, 'popularity'] = np.linspace(5, 1, length)
            interaction = np.array(self.trainData['popularity'])
            rows = np.array(self.trainData['playlist_id'])
            cols = np.array(self.trainData['track_id'])
            self.URM_all = sps.csr_matrix((interaction, (rows, cols)))
        else:
            grouped = self.trainData.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))
            matrix = MultiLabelBinarizer(classes=self.get_tracks(), sparse_output=True).fit_transform(grouped)
            self.URM_all = matrix.tocsr()
        print("PlaylistDataReader: URM matrix built completed")
        print("PlaylistDataReader: shape is {}".format(self.URM_all.shape))
        return

    def build_UCM(self):
        print("PlaylistDataReader: UCM Matrix is being built...")
        UCM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(self.URM_all.T)
        UCM_tfidf = UCM_tfidf.T
        self.UCM = UCM_tfidf
        print("PlaylistDataReader: UCM matrix built completed")
        print("PlaylistDataReader: shape is {}".format(self.UCM.shape))
        return

    def build_ICM(self):
        print("PlaylistDataReader: ICM Matrix is being built...")
        # artists
        df_artists = self.trackData.reindex(columns=["track_id", "artist_id"])
        df_artists.sort_values(by="track_id", inplace=True)
        artist_list = [[a] for a in df_artists["artist_id"]]
        icm_artists = MultiLabelBinarizer(
            classes=self.get_artists(), sparse_output=True).fit_transform(artist_list)
        icm_artists_csr = icm_artists.tocsr()
        # albums
        df_albums = self.trackData.reindex(columns=["track_id", "album_id"])
        df_albums.sort_values(by="track_id", inplace=True)
        album_list = [[b] for b in df_albums["album_id"]]
        icm_albums = MultiLabelBinarizer(
            classes=self.get_albums(), sparse_output=True).fit_transform(album_list)
        icm_albums_csr = icm_albums.tocsr()
        ICM = sps.hstack((icm_artists_csr, icm_albums_csr))
        ICM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(ICM)
        ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')
        self.ICM = ICM_tfidf.tocsr()
        print("PlaylistDataReader: ICM matrix built completed")
        print("PlaylistDataReader: shape is {}".format(self.ICM.shape))
        return

    def split(self,divide_two = False):
        print("PlaylistDataReader: URM_train and URM_test are being built...")
        self.URM_all = self.URM_all.tocoo()
        numInteractions = len(self.URM_all.data)
        df_test = []
        split_mask = []
        # Divide the test set in half
        test_set_length = len(self.targetData.playlist_id.values)
        divider_mask = shuffle(np.array([True] * (int(test_set_length / 2)) + [False] * int(test_set_length / 2)),
                               random_state=666)
        divider_mask = np.logical_not(divider_mask)
        test_list = self.targetData.playlist_id.values[divider_mask]
        for p in self.targetData.playlist_id.values:
            values = self.trainData[self.trainData.playlist_id == p].index
            # guarantee to get songs from every playlist
            masked = np.random.choice([True,False],len(values),p=[0.2,0.8])
            while(sum(masked) == 0):
                masked = np.random.choice([True,False],len(values),p=[0.2,0.8])
            split_mask.extend(masked)
            df_test.extend(values)
        df_test = np.array(df_test).ravel()
        split_mask = np.array(split_mask).ravel()
        #split_mask = np.random.choice([True, False], len(df_test), p=[0.2, 0.8])
        df_test_final = df_test[split_mask]
        test_mask = np.zeros((numInteractions, 1), dtype=bool).ravel()
        test_mask[df_test_final] = True
        self.URM_test = sps.coo_matrix((self.URM_all.data[test_mask], (self.URM_all.row[test_mask], self.URM_all.col[test_mask])))
        train_mask = np.logical_not(test_mask)
        self.URM_train = sps.coo_matrix((self.URM_all.data[train_mask], (self.URM_all.row[train_mask], self.URM_all.col[train_mask])))
        if divide_two:
            # Delete the songs of half of the playlists
            self.URM_test = self.URM_test.todense()
            self.URM_test[test_list,:] = 0
            self.URM_test = sps.coo_matrix(self.URM_test)
        # print("PlaylistDataReader: saving URM_train and URM_test")
        # sps.save_npz(self.dataSubfolder + "URM_train.npz", self.URM_train)
        # sps.save_npz(self.dataSubfolder + "URM_test.npz", self.URM_test)
        if self.verbose:
            print("PlaylistDataReader: Data loading is complete")
        return

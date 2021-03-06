

import numpy as np
import scipy.sparse as sps
import pandas as pd
from sklearn import feature_extraction
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.utils import shuffle

from base.RecommenderUtils import to_okapi


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
        self.dataSubfolder = "./data/"
        self.trainData = pd.read_csv(self.dataSubfolder + "train.csv")
        self.trackData = pd.read_csv(self.dataSubfolder + "tracks.csv")
        self.targetData = pd.read_csv(self.dataSubfolder + "target_playlists.csv")
        self.trainSeq = pd.read_csv(self.dataSubfolder + "train_sequential.csv")
        self.artists = sps.load_npz(self.dataSubfolder + "artists.npz")
        self.albums = sps.load_npz(self.dataSubfolder + "albums.npz")
        # Add each track.playlist to its original index
        self.trainData["index"] = self.trainData.index
        # merge the indexes
        self.trainSeq = self.trainSeq.merge(self.trainData, on=["playlist_id", "track_id"])
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

    def get_target_playlists_seq(self):
        target_playlists= self.targetData['playlist_id'].unique()[:5000]
        return np.sort(target_playlists)

    def get_artists(self):
        artists = self.trackData['artist_id'].unique()
        return np.sort(artists)

    def get_albums(self):
        albums = self.trackData['album_id'].unique()
        return np.sort(albums)

    def get_URM_train(self):
        if self.URM_train is None:
            raise TypeError("PlaylistDataReader: URM_train is not built")
        else:
            return self.URM_train

    def get_URM_test(self):
        if self.URM_test is None:
            raise TypeError("PlaylistDataReader: URM_test is not built")
        else:
            return self.URM_test
    def get_URM_all(self):
        if self.URM_all is None:
            raise TypeError("PlaylistDataReader: URM is not built")
        else:
            return self.URM_all

    def get_ICM(self):
        if self.ICM is None:
            raise TypeError("PlaylistDataReader: ICM is not built")
        else:
            return self.ICM

            
    def get_UCM(self):
        if self.UCM is None:
            raise TypeError("PlaylistDataReader: UCM is not built")
        else:
            return self.UCM

    def get_URM_train_tfidf(self):
        if self.URM_train is None:
            raise TypeError("PlaylistDataReader: URM train is not build")
        else:
            self.URM_train_t = self.URM_train.T
            URM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(self.URM_train_t)
            URM_tfidf = URM_tfidf_T.T
            self.URM_train_tfidf = URM_tfidf.tocsr()
            return self.URM_train_tfidf

    def get_URM_train_okapi(self):
        if self.URM_train is None:
            raise TypeError("PlaylistDataReader: URM train is not build")
        else:
            self.URM_train_okapi = to_okapi(self.URM_train)
            return self.URM_train_okapi


    def get_UCM_tfidf(self):
        if self.UCM is None:
            raise TypeError("PlaylistDataReader: UCM is not built")
        else:
            self.UCM_t = self.UCM.T
            UCM_tfidf_t = feature_extraction.text.TfidfTransformer().fit_transform(self.UCM_t)
            UCM_tfidf = UCM_tfidf_t.T
            self.UCM_tfidf = UCM_tfidf.tocsr()

    def build_URM(self):
        #print("PlaylistDataReader: URM Matrix is being built...")
        if self.loadPredefined:
            try:
                self.URM_all = sps.load_npz(self.dataSubfolder + "URM_all.npz")
                print("PlaylistDataReader: URM is imported from the saved data.")
                return
            except FileNotFoundError:
                print("PlaylistDataReader: URM is not.Found. Building new one")
                grouped = self.trainData.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))
                matrix = MultiLabelBinarizer(classes=self.get_tracks(), sparse_output=True).fit_transform(grouped)
                self.URM_all = matrix.tocsr()
                sps.save_npz(self.dataSubfolder + "URM_all.npz", self.URM_all)
                print("PlaylistDataReader: URM matrix built completed")
                print("PlaylistDataReader: shape is {}".format(self.URM_all.shape))
        return
    # UCM matrix + tfidf
    def build_UCM(self):
        #print("PlaylistDataReader: UCM Matrix is being built...")
        stack = sps.hstack((self.artists, self.albums)).tocsr()
        UCM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(stack.T)
        UCM_tfidf = UCM_tfidf.T
        UCM_tfidf = normalize(UCM_tfidf,axis=0,norm='l2')
        self.UCM = UCM_tfidf
        print("PlaylistDataReader: UCM matrix built completed")
        print("PlaylistDataReader: shape is {}".format(self.UCM.shape))
        return

    def build_ICM(self):
        if self.loadPredefined:
            try:
                self.ICM = sps.load_npz(self.dataSubfolder + "ICM.npz")
                print("PlaylistDataReader: ICM is imported from the saved data.")
                return
            except FileNotFoundError:
                print("PlaylistDataReader: ICM is not.Found. Building new one")
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
                # ICM_tfidf_T = feature_extraction.text.TfidfTransformer().fit_transform(ICM.T)
                # ICM_tfidf = ICM_tfidf_T.T
                # ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')
                # self.ICM = ICM_tfidf.tocsr()
                self.ICM = ICM
                sps.save_npz(self.dataSubfolder + "ICM.npz", self.ICM)
                print("PlaylistDataReader: ICM matrix built completed")
                print("PlaylistDataReader: shape is {}".format(self.ICM.shape))
            
        return

    def split(self,divide_two=False):
        if self.loadPredefined:
            try:
                self.URM_train = sps.load_npz(self.dataSubfolder + "URM_train.npz")
                self.URM_test = sps.load_npz(self.dataSubfolder + "URM_test.npz")
                print("PlaylistDataReader: URM train and test splits are imported from the saved data.")
                return
            except FileNotFoundError:
                print("PlaylistDataReader: URM_train or URM_test not.Found. Building new ones")
        else:
            print("PlaylistDataReader: URM_train and URM_test are being built...")
            self.URM_all = self.URM_all.tocoo()
            numInteractions = len(self.URM_all.data)
            df_test = []
            split_mask = []
            # Divide the test set in half
            test_set_length = len(self.targetData.playlist_id.values)
            divider_mask = shuffle(np.array([True] * (int(test_set_length / 2)) + [False] * int(test_set_length / 2)),random_state=0)
            divider_mask = np.logical_not(divider_mask)
            test_list = self.targetData.playlist_id.values[divider_mask]
            # # Create the test set
            self.sequential_ones = self.get_target_playlists_seq()
            if self.adjust:
                for p in self.targetData.playlist_id.values:
                    if p in self.sequential_ones:
                        values = self.trainSeq[self.trainSeq.playlist_id == p]["index"].values
                        seq_val = int(len(values) * 0.2)
                        masked = np.array([[False] * (len(values) - seq_val) + [True] * seq_val]).ravel()
                    else:
                        values = self.trainData[self.trainData.playlist_id == p].index
                    # guarantee to get songs from every playlist
                        masked = np.random.choice([True,False],len(values),p=[0.2,0.8])
                        while(sum(masked) == 0):
                            masked = np.random.choice([True,False],len(values),p=[0.2,0.8])
                    split_mask.extend(masked)
                    df_test.extend(values)
            else:
                for p in self.targetData.playlist_id.values:
                    values = self.trainData[self.trainData.playlist_id == p].index
                # guarantee to get songs from every playlist
                    masked = np.random.choice([True, False], len(values), p=[0.2, 0.8])
                    while(sum(masked) == 0):
                        masked = np.random.choice([True, False], len(values), p=[0.2, 0.8])
                    split_mask.extend(masked)
                    df_test.extend(values)
            df_test = np.array(df_test).ravel()
            split_mask = np.array(split_mask).ravel()
            #split_mask = np.random.choice([True, False], len(df_test), p=[0.2, 0.8])
            df_test_final = df_test[split_mask]
            test_mask = np.zeros((numInteractions, 1), dtype=bool).ravel()
            test_mask[df_test_final] = True
            ints = self.URM_all.data[test_mask]
            rows = self.URM_all.row[test_mask]
            cols = self.URM_all.col[test_mask]
            self.URM_test = sps.coo_matrix((ints, (rows, cols)))
            train_mask = np.logical_not(test_mask)
            self.URM_train = sps.coo_matrix((self.URM_all.data[train_mask], (self.URM_all.row[train_mask], self.URM_all.col[train_mask])))
            if divide_two:
                # Delete the songs of half of the playlists
                self.URM_test = self.URM_test.todense()
                self.URM_test[test_list,:] = 0
                self.URM_test = sps.coo_matrix(self.URM_test)
            print("PlaylistDataReader: saving URM_train and URM_test")
            sps.save_npz(self.dataSubfolder + "URM_train.npz", self.URM_train)
            sps.save_npz(self.dataSubfolder + "URM_test.npz", self.URM_test)
        return

    def save_dataset(self):
        print("PlaylistDataReader: saving URM_train and URM_test")
        sps.save_npz(self.dataSubfolder + "URM_train.npz", self.URM_train)
        sps.save_npz(self.dataSubfolder + "URM_test.npz", self.URM_test)

    def generate_datasets(self):
        self.build_URM()
        self.build_ICM()
        self.split()
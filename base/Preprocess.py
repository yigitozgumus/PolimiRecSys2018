import numpy as np
import pandas as pd
import scipy.sparse as sps
from base.RecommenderUtils import check_matrix


class Preprocess(object):
    def __init__(self,
                 trainData,
                 trackData,
                 tracks=False,
                 albums=False,
                 artists=False,
                 normalize=False):
        super(Preprocess, self).__init__()
        self.tracks = tracks
        self.albums = albums
        self.artists = artists
        self.trainData = trainData
        self.trackData = trackData
        self.normalize = normalize

    def pipeline(self):
        ICM = self.trackData
        lenTracks = 0
        lenAlbums = 0
        lenArtist = 0
        if self.tracks:
            trackCounts = self.trainData['track_id'].value_counts().reset_index().rename(
                columns={"index": "track_id", "track_id": "track_counts"})
            trackCounts = trackCounts.sort_values('track_counts', ascending=False)
            if self.normalize:
                lenTracks = len(trackCounts)
                #trackCounts.track_counts / len(trackCounts)
            ICM = pd.merge(ICM, trackCounts, on='track_id')
            # featureMatrix = featureMatrix.drop(['duration_sec', 'album_id', 'artist_id'], axis=1)
        if self.albums:
            albumCounts = self.trackData['album_id'].value_counts().reset_index().rename(
                columns={"index": "album_id", "album_id": "album_counts"})
            albumCounts = albumCounts.sort_values('album_counts', ascending=False)
            if self.normalize:
                lenAlbums = len(albumCounts)
                #albumCounts.album_counts /= len(albumCounts)
            # add normalization
            ICM = pd.merge(ICM, albumCounts, on='album_id')
        if self.artists:
            artistCounts = self.trackData['artist_id'].value_counts().reset_index().rename(
                columns={"index": "artist_id", "artist_id": "artist_counts"})
            artistCounts = artistCounts.sort_values("artist_counts", ascending=False)
            if self.normalize:
                lenArtist = len(artistCounts)
                #artistCounts.artist_counts /= len(artistCounts)
            ICM = pd.merge(ICM, artistCounts, on="artist_id")
       # ICM["counts"] = (ICM.track_counts + ICM.album_counts + ICM.artist_counts) / (lenTracks + lenAlbums + lenArtist)
        ICM = ICM.drop(["duration_sec", "album_id", "artist_id"], axis=1)
       # ICM = ICM.drop(["artist_counts","track_counts","album_counts"], axis=1)
        #ICM.counts *= 100
        ICM = sps.coo_matrix(ICM)
        # add to coo matrix
        return ICM

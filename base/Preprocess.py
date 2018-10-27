import numpy as np
import pandas as pd
import scipy.sparse as sps
from base.RecommenderUtils import check_matrix


def extractTrackPopularity(trainData,trackData):
    trackCounts = trainData['track_id'].value_counts().reset_index().rename(columns={"index":"track_id","track_id":"track_counts"})
    trackCounts = trackCounts.sort_values('track_counts',ascending=False)
    featureMatrix =  pd.merge(trackData,trackCounts,on='track_id')
    featureMatrix = sps.coo_matrix(featureMatrix)
    ICM = check_matrix(featureMatrix,'csr')
    return ICM


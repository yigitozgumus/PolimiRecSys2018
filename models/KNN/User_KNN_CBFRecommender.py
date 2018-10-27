import numpy as np

from base.BaseRecommender import RecommenderSystem
from base.BaseRecommender_SM import RecommenderSystem_SM
from base.RecommenderUtils import check_matrix
from base.Similarity import Similarity

class UserKNNCBFRecommender(RecommenderSystem, RecommenderSystem_SM):
    
    def __init__(self,URM_train,
                ICM,
                sparse_weights= True,
                verbose= False,
                similarity_mode="cosise"):
        super(UserKNNCBFRecommender,self).__init__()
        self.URM_train = check_matrix(URM_train,'csr')
        self.ICM = ICM
        self.sparse_weights = sparse_weights
        self.verbose = verbose
        self.similarity_mode = similarity_mode
        self.parameters = None

        

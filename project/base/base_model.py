class BaseModel(object):
    def __init(self,config):
        self.config = config
        self.train_data = None
        self.track_data = None
        self.target = None
    
    def fit(self,URM_train):
         raise NotImplementedError

    def recommend(self,playlist_id,at=10):
         raise NotImplementedError

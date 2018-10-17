class BaseModel(object):
    def __init(self):
        self.train_data = None
        self.track_data = None
        self.target = None
    
    def fit(self):
         raise NotImplementedError

    def recommend(self):
         raise NotImplementedError

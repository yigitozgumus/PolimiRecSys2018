
"""

@author: Semsi Yigit Ozgumus
"""


class BaseRecommender(object):
    def __init(self):
        self.train_data = None
        self.track_data = None
        self.target = None
    
    def fit(self):
         pass

    def recommend(self):
         raise NotImplementedError

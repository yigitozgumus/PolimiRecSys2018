from base.base_model import BaseModel

class TopPopRecommender(BaseModel):

    def __init__(self,data):
        super(TopPopRecommender,self).__init__(data)
        
    
    def fit(self, URM_train):

        itemPopularity = (URM_train>0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis = 0)
    
    
    def recommend(self, playlist_id, at=10):   
        recommended_items = self.popularItems[0:at]
        return recommended_items


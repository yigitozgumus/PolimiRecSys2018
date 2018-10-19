#from TopPop.top_pop_recommender import TopPopRecommender
from base.Similarity import Similarity
from project.project import RecommenderSystem
# from utils.config import process_config
from models.TopPop.top_pop_recommender import TopPopRecommender
from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from data.PlaylistDataReader import PlaylistDataReader
from utils.export import export_submission


def main():
    # Load the data
    dataReader = PlaylistDataReader()
    # Prepare the models
    recsys = UserKNNCFRecommender(dataReader.URM_train)
    recsys.fit(shrink=100)
    # make prediction
    recsys.evaluateRecommendations(dataReader.URM_test,at=10,exclude_seen=True)
    # export the best one
    export_submission(dataReader.targetData, recsys)



if __name__ == "__main__":
    main()

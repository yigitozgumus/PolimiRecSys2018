#from TopPop.top_pop_recommender import TopPopRecommender

# from utils.config import process_config

from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from data.PlaylistDataReader import PlaylistDataReader
from utils.export import Logger
from utils.config import clear

def main():
    clear()

    # Load the data
    dataReader = PlaylistDataReader()
    l = Logger(dataReader.targetData)

    # Prepare the models
    recsys = []
    recsys.append(UserKNNCFRecommender(dataReader.URM_train,verbose=True,similarity_mode="pearson"))

    # Train the models
    for model in recsys:
        model.fit()

    # make prediction
    for model in recsys:
        model.evaluateRecommendations(dataReader.URM_test,at=10,exclude_seen=True)


    # export the predictions
    l.export_submissions(recsys)
    l.log_experiment()




if __name__ == "__main__":
    main()

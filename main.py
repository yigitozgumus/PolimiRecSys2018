#from TopPop.top_pop_recommender import TopPopRecommender

# from utils.config import process_config

from models.KNN.User_KNN_CFRecommender import UserKNNCFRecommender
from models.KNN.Item_KNN_CFRecommender import ItemKNNCFRecommender
from data.PlaylistDataReader import PlaylistDataReader
from utils.logger import Logger
from utils.config import clear, Configurator
import argparse



def main():
    parser = argparse.ArgumentParser(description="Main runner script of the Recommendation System Project")
    parser.add_argument("json", metavar="J", help="Target Location of the config file")
    args = parser.parse_args()
    fileName = args.json
    pipeline(fileName)

def pipeline(fileName):
    clear()
    # Load the data
    data_reader = PlaylistDataReader(verbose=True)
    l = Logger(data_reader.targetData)
    conf = Configurator(fileName,data_reader)
    # Prepare the models
    rec_sys = conf.extractModels()
    # Train the models
    for model in rec_sys:
        model.fit()

    # make prediction
    for model in rec_sys:
        model.evaluateRecommendations(
            data_reader.URM_test, at=10, exclude_seen=True)

    # export the predictions
    l.export_submissions(rec_sys)
    l.log_experiment()


if __name__ == "__main__":
    main()

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
    parser.add_argument("--json", "-j", metavar="J", help="Target Location of the config file")
    parser.add_argument("--export", "-e",help="Whether you want to export it or not", action='store_false', default=True,dest="exp_switch")
    parser.add_argument("--log", "-l", help="Whether you want to log it or not",
                        action='store_false', default=True, dest="log_switch")
    args = parser.parse_args()
    fileName = args.json
   # print(args.json)
   # print(args.exp_switch)
   # print(args.log_switch)
    pipeline(fileName,args.exp_switch,args.log_switch)

def pipeline(fileName,exp_,log_):
    clear()
    # Load the data
    conf = Configurator(fileName)
    
    data_reader = PlaylistDataReader(verbose=False,)
    l = Logger(data_reader.targetData)
    
    # Prepare the models
    rec_sys = conf.extractModels(data_reader)
    
    for model in rec_sys:
        # Train the models
        model.fit()
        # make prediction
        model.evaluateRecommendations(data_reader.URM_test,at=10, exclude_seen=True)

    # export the predictions
    if exp_:
        l.export_submissions(rec_sys)
    if log_:
        l.log_experiment()



if __name__ == "__main__":
    main()

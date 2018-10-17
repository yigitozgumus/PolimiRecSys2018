from project.project import RecommenderSystem
#from utils.config import process_config
from models.top_pop_recommender_unseen import TopPopRecommenderUnseen
from data.PlaylistDataReader import PlaylistDataReader
from utils.export import export_submission

def main():
   # config = process_config("configs/initial_config.json")
    dataReader = PlaylistDataReader()
    #print(dataReader)
    recsys= RecommenderSystem(dataReader,TopPopRecommenderUnseen)
    recsys.pipeline()
    export_submission(dataReader.targetData,recsys.model)

if __name__ == "__main__":
    main()

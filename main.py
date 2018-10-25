from data.PlaylistDataReader import PlaylistDataReader
from utils.logger import Logger
from utils.config import clear, Configurator
import argparse


def main():
    parser = argparse.ArgumentParser(description="Main runner script of the Recommendation System Project")
    parser.add_argument("--json", "-j", metavar="J", help="Target Location of the config file")
    parser.add_argument("--export", "-e", help="Whether you want to export it or not", action='store_false',
                        default=True, dest="exp_switch")
    parser.add_argument("--log", "-l", help="Whether you want to log it or not",
                        action='store_false', default=True, dest="log_switch")
    args = parser.parse_args()
    file_name = args.json
    pipeline(file_name, args.exp_switch, args.log_switch)


def pipeline(fileName, exp_, log_):
    clear()
    # Load the data
    conf = Configurator(fileName)

    data_reader = PlaylistDataReader(adjustSequentials=True)
    l = Logger(data_reader.targetData)
    # Prepare the models
    rec_sys = conf.extract_models(data_reader)
    for model in rec_sys:
        # Train the models
        model.fit()
        # make prediction
        model.evaluate_recommendations(data_reader.URM_test, at=10, exclude_seen=True)

    # export the predictions
    if exp_:
        l.export_experiments(rec_sys)
    if log_:
        l.log_experiment()


if __name__ == "__main__":
    main()

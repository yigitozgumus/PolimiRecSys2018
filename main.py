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
    parser.add_argument("--logFile","-lf",metavar="L", help="To open a specific log file to track specific experiment")
    args = parser.parse_args()
    file_name = args.json
    logFile = None
    if not args.logFile is None:
        logFile = args.logFile
    pipeline_stable(file_name, args.exp_switch, args.log_switch, logFile)
    #pipeline_dev(file_name, args.exp_switch, args.log_switch,logFile)


def pipeline_stable(fileName, exp_, log_, logFile):
    clear()
    # Load the data
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader(adjustSequentials=conf.configs.dataReader["adjustSequentials"])
    l = Logger(data_reader.targetData, logFile)
    # Prepare the models
    rec_sys = conf.extract_models(data_reader)

    for model in rec_sys:
        model.fit() # Train the models
        model.evaluate_recommendations(data_reader.URM_test, at=10, exclude_seen=True) # make prediction
    #export the predictions
    if exp_:
        l.export_experiments(rec_sys)
    if log_:
        l.log_experiment()

def pipeline_dev(fileName, exp_, log_, logFile):
    clear()
    # Load the data
    conf = Configurator(fileName)

    data_reader = PlaylistDataReader(adjustSequentials=conf.configs.dataReader["adjustSequentials"])
    l = Logger(data_reader.targetData,logFile)
    # Prepare the models
    rec_sys = conf.extract_models(data_reader)
    # Shrink exp
    for method in conf.configs.sgd_mode:
        for model in rec_sys:
            model.fit(epochs=method)  # Train the models
            model.evaluate_recommendations(
                data_reader.URM_test, at=10, exclude_seen=True)  # make prediction
        if exp_:
            l.export_experiments(rec_sys)
        if log_:
            l.log_experiment()


if __name__ == "__main__":
    main()

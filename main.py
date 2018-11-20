
from data.PlaylistDataReader import PlaylistDataReader
from utils.logger import Logger
from utils.config import clear, Configurator
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Main runner script of the Recommendation System Project")
    parser.add_argument("--json", "-j", metavar="J",
                        help="Target Location of the config file (relative to script directory)")
    parser.add_argument("--mode", "-m", metavar="M",
                        help="Choose between stable, dev, or mf(1, 2 or 3 respectively)", type=int)
    parser.add_argument("--export", "-e", help="Whether you want to export it or not (add it to disable)", action='store_false',
                        default=True, dest="exp_switch")
    parser.add_argument("--log", "-l", help="Whether you want to log it or not (add it to disable)",
                        action='store_false', default=True, dest="log_switch")
    parser.add_argument("--logFile", "-lf", metavar="L",
                        help="To export a specific log file to track specific experiment (default is the Logs.md)")

    args = parser.parse_args()
    file_name = args.json
    logFile = None
    mode = args.mode
    if not args.logFile is None:
        logFile = args.logFile
    if mode == 1:
        pipeline_stable(file_name, args.exp_switch, args.log_switch, logFile)
    elif mode == 2:
        pipeline_dev(file_name, args.exp_switch, args.log_switch, logFile)
    elif mode == 3:
        pipeline_mf(file_name, args.exp_switch, args.log_switch, logFile)
    elif mode == 4:
        pipeline_submission(file_name, args.exp_switch, args.log_switch, logFile)


def pipeline_mf(fileName, exp_, log_, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader(
        adjustSequentials=conf.configs.dataReader["adjustSequentials"])
    data_reader.build_URM()
    data_reader.build_UCM()
    data_reader.build_ICM()
    data_reader.split()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader)
    for model in rec_sys:
        model.fit(data_reader.get_URM_train())  # Train the models
        model.evaluate_recommendations(
            data_reader.URM_test, at=10, exclude_seen=True)  # make prediction
    if exp_:
        l.export_experiments(rec_sys)
    if log_:
        l.log_experiment()


def pipeline_stable(fileName, exp_, log_, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.build_URM()
    data_reader.build_UCM()
    data_reader.build_ICM()
    data_reader.split()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader)
    for model in rec_sys:
        model.fit()
        model.evaluate_recommendations(
            data_reader.get_URM_test(), at=10, exclude_seen=True)  # make prediction
    if exp_:
        l.export_experiments(rec_sys)
    if log_:
        l.log_experiment()


def pipeline_dev(fileName, exp_, log_, logFile):
    clear()
    # Load the data
    conf = Configurator(fileName)

    data_reader = PlaylistDataReader()
    data_reader.build_URM()
    data_reader.build_UCM()
    data_reader.build_ICM()
    data_reader.split()
    l = Logger(data_reader.targetData, logFile)
    # Prepare the models
    rec_sys = conf.extract_models(data_reader)
    # Shrink exp

    #for k in conf.configs.lambda_j:
    for k in conf.configs.gamma:
        for model in rec_sys:
            model.fit(gamma = k)  # Train the models
            #model.fit(lambda_j=k)
            model.evaluate_recommendations(data_reader.URM_test, at=10, exclude_seen=True)  # make prediction
            if exp_:
                l.export_experiments(rec_sys)
            if log_:
                l.log_experiment()

def pipeline_submission(fileName,exp_,log_,logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.build_URM()
    data_reader.build_UCM()
    data_reader.build_ICM()
    data_reader.split()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader,submission=True)
    for model in rec_sys:
        model.fit()
        model.evaluate_recommendations(data_reader.get_URM_test(), at=10, exclude_seen=True)  # make prediction
    if exp_:
        l.export_experiments(rec_sys)
    if log_:
        l.log_experiment(submission=True)

if __name__ == "__main__":
    main()

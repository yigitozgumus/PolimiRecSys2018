
from data.PlaylistDataReader import PlaylistDataReader
from utils.logger import Logger
from utils.config import clear, Configurator
import argparse
from base.evaluation.Evaluator import SequentialEvaluator



def main():
    parser = argparse.ArgumentParser(
        description="Main runner script of the Recommendation System Project")
    parser.add_argument("--json", "-j", metavar="J",
                        help="Target Location of the config file (relative to script directory)")
    parser.add_argument("--mode", "-m", metavar="M",
                        help="Choose between stable, dev, or mf(1, 2 or 3 respectively)", type=int)
    parser.add_argument("--logFile", "-lf", metavar="L",
                        help="To export a specific log file to track specific experiment (default is the Logs.md)")

    args = parser.parse_args()
    file_name = args.json
    logFile = None
    mode = args.mode
    if not args.logFile is None:
        logFile = args.logFile
    if mode == 1:
        pipeline_stable(file_name, logFile)
    elif mode == 2:
        print("Deprecated. Use another mode.")
    elif mode == 3:
        print("Deprecated. Use another mode.")
    elif mode == 4:
        pipeline_submission(file_name, logFile)
    elif mode == 5:
        pipeline_parameter_tuning()


def pipeline_parameter_tuning():
    pass



def pipeline_stable(fileName, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.build_URM()
    data_reader.build_UCM()
    data_reader.build_ICM()
    data_reader.split()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader)
    evaluator = SequentialEvaluator(data_reader.get_URM_test(), [10], exclude_seen=True)
    for model in rec_sys:
        model.fit()
        results_run, results_run_string = evaluator.evaluateRecommender(model)
        print("Algorithm: {}, results: \n{}".format(str(model), results_run_string))
        l.export_experiments(model,results_run_string)
        l.log_experiment()



def pipeline_submission(fileName,logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.build_URM()
    data_reader.build_UCM()
    data_reader.build_ICM()
    data_reader.split()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader,submission=True)
    evaluator = SequentialEvaluator(data_reader.get_URM_all(), [10], exclude_seen=True)
    for model in rec_sys:
        model.fit()
        results_run, results_run_string = evaluator.evaluateRecommender(model)
        print("Algorithm: {}, results: \n{}".format(str(model), results_run_string))
        l.export_experiments(rec_sys,results_run_string)
        l.log_experiment(submission=True)

if __name__ == "__main__":
    main()


from data.PlaylistDataReader import PlaylistDataReader
from utils.logger import Logger
from utils.config import clear, Configurator
import argparse
from base.evaluation.Evaluator import SequentialEvaluator
from utils.tuner import read_data_split_and_search





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
        pipeline_best(file_name, logFile)
    elif mode == 3:
        read_data_split_and_search()
    elif mode == 4:
        pipeline_save_training_model(file_name,logFile)
    elif mode == 5:
        pipeline_submission_hybrid(file_name, logFile)
    elif mode == 6:
        pipeline_save_model(file_name,logFile)


def pipeline_save_training_model(fileName, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.generate_datasets()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader)
    evaluator = SequentialEvaluator(data_reader.get_URM_test(), [10], exclude_seen=True)
    for model in rec_sys:
        model.fit(save_model=True,best_parameters=False)
        results_run, results_run_string = evaluator.evaluateRecommender(model)
        print("Algorithm: {}, results: \n{}".format(str(model), results_run_string))
        l.export_experiments(model, results_run_string)
        l.log_experiment()

def pipeline_save_model(fileName, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.generate_datasets()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader)
    evaluator = SequentialEvaluator(data_reader.get_URM_all(), [10], exclude_seen=True)
    for model in rec_sys:
        model.fit(save_model=True,best_parameters=True)
        #results_run, results_run_string = evaluator.evaluateRecommender(model)
        #print("Algorithm: {}, results: \n{}".format(str(model), results_run_string))
        #l.export_experiments(model, results_run_string)
        #l.log_experiment()

def pipeline_stable(fileName, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.generate_datasets()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader)
    evaluator = SequentialEvaluator(data_reader.get_URM_test(), [10], exclude_seen=True)
    for model in rec_sys:
        model.fit()
        results_run, results_run_string = evaluator.evaluateRecommender(model)
        print("Algorithm: {}, results: \n{}".format(str(model), results_run_string))
        l.export_experiments(model,results_run_string)
        l.log_experiment()

def pipeline_best(fileName, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.generate_datasets()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader)
    evaluator = SequentialEvaluator(data_reader.get_URM_test(), [10], exclude_seen=True)
    for model in rec_sys:
        model.fit(best_parameters=True)
        results_run, results_run_string = evaluator.evaluateRecommender(model)
        print("Algorithm: {}, results: \n{}".format(str(model), results_run_string))
        l.export_experiments(model,results_run_string)
        l.log_experiment()



def pipeline_submission_hybrid(fileName, logFile):
    clear()
    conf = Configurator(fileName)
    data_reader = PlaylistDataReader()
    data_reader.generate_datasets()
    l = Logger(data_reader.targetData, logFile)
    rec_sys = conf.extract_models(data_reader,submission=True)
    evaluator = SequentialEvaluator(data_reader.get_URM_all(), [10], exclude_seen=True)
    for model in rec_sys:
        model.fit(submission=True,best_parameters=True)
    #    results_run, results_run_string = evaluator.evaluateRecommender(model)
    #    print("Algorithm: {}, results: \n{}".format(str(model), results_run_string))
        result_string = "Full Submission No MAP"
        l.export_experiments(model,result_string)
        l.log_experiment(submission=True)

if __name__ == "__main__":
    main()

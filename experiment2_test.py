import pandas as pd
import numpy as np
import diagnostic_model
import multiprocessing
from queue import Queue

def LR_trial1():
  # Test run --> Logistic Regression algorithm w/out kFold
  try:
      diagnostic_model.test_diagnostic_model(derivative="rois_aal", strategy="filt_global", pipeline="cpac", algorithm="LR", print_stats=True, kFold=False)
  except Exception as e:
      print(f"Error in model execution: {e}")

def LR_trial2():
  # Test run --> Logistic Regression algorithm w/ kFold
  try:
      diagnostic_model.test_diagnostic_model(derivative="rois_aal", strategy="filt_global", pipeline="cpac", algorithm="LR", print_stats=True, kFold=True)
  except Exception as e:
      print(f"Error in model execution: {e}")

def XGB_trial1():
  # Test run --> XGBoost algorithm w/out kFold
  try:
    diagnostic_model.test_diagnostic_model(derivative="rois_aal", strategy="filt_global", pipeline="cpac", algorithm="XGB", print_stats=True, kFold=False)
  except Exception as e:
    print(f"Error in model execution: {e}")

def XGB_trial2():
  # Test run --> XGBoost algorithm w/ kFold
  try:
    diagnostic_model.test_diagnostic_model(derivative="rois_aal", strategy="filt_global", pipeline="cpac", algorithm="XGB", print_stats=True, kFold=True)
  except Exception as e:
    print(f"Error in model execution: {e}")

# Modify process_combination to use Queue
def process_combination(derivative, pipeline, strategy, algorithms, oversampler, padder, model_count, queue):
    try:
        # Download data
        diagnostic_model.download_data(desired_derivative=derivative, desired_strategy=strategy, desired_pipeline=pipeline, print_stats=False)
        print(f"Downloaded data for parameters -> derivative: {derivative}, pipeline: {pipeline}, strategy: {strategy}")
    except Exception as e:
        print(f"Error in data downloading: {e}")
    
    try:
        # Extract features and labels
        features, labels = diagnostic_model.features_and_labels(derivative=derivative, pipeline=pipeline, strategy=strategy, filler=padder, print_stats=False, oversampler=oversampler)
        print()
        print(f"Extracted features and labels -> filler value: {padder}, oversampler: {oversampler}")

        for algorithm in algorithms:
            try:
                # Run the model
                model_performance = diagnostic_model.train_test_fMRI_data_kfold(fMRI_features=features, labels=np.ravel(labels), algorithm=algorithm, k=5, print_stats=False, algorithm_hypertuned=False)

                model_data = [str(item) for item in [model_count, algorithm, derivative, pipeline, strategy, oversampler, padder] + model_performance[1:5] + [model_performance[0]] + model_performance[5:]]
                # Put the model's performance metrics into the queue
                queue.put(model_data)
            except Exception as e:
                print(f"Error in execution of Model-{model_count}: {e}")
            print(f"Model-{model_count}'s testing has been completed!")
            model_count += 1
    except Exception as e:
        print(f"Error in processing combination: {e}")

def main_test_multiprocessing():
    derivatives = ["rois_aal", "rois_cc200", "rois_cc400", "rois_dosenbach160", "rois_ez", "rois_ho", "rois_tt"]
    algorithms = ["LR", "XGB"]
    pipelines = ["cpac", "niak"]
    strategies = ["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"]
    oversamplers = [None, "Random", "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE"]
    padding_methods = ["zero", "mean", "median"]

    column_titles = ["Model #", "ML Algorithm", "Derivative", "Preprocessing Pipeline", "Preprocessing Strategy", "Oversampler", "Padding Method",
                    "Sensitivity", "Specificity", "Precision", "F1 Score", "Accuracy",
                    "True Positives", "True Negatives", "False Positives", "False Negatives"]
    experiment_data = pd.DataFrame(columns=column_titles)
    model_count = 1

    # Create a Queue to store the results
    queue = Queue()

    jobs = []
    for derivative in derivatives:
        for pipeline in pipelines:
            for strategy in strategies:
                for oversampler in oversamplers:
                    for padder in padding_methods:
                        p = multiprocessing.Process(target=process_combination, args=(derivative, pipeline, strategy, algorithms, oversampler, padder, model_count, queue))
                        jobs.append(p)
                        p.start()

    # Retrieve results from the Queue and put them into the DataFrame
    while len(experiment_data) < len(derivatives) * len(pipelines) * len(strategies) * len(oversamplers) * len(padding_methods) * len(algorithms):
        model_data = queue.get()
        experiment_data.loc[len(experiment_data)] = model_data

    return experiment_data

def main_test_singleprocessing():
    derivatives = ["rois_aal", "rois_cc200", "rois_cc400", "rois_dosenbach160", "rois_ez", "rois_ho", "rois_tt"]
    algorithms = ["LR", "XGB"]
    pipelines = ["cpac", "niak"]
    strategies = ["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"]
    oversamplers = [None, "Random", "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE"]
    padding_methods = ["zero", "mean", "median"]
    model_count = 1
    column_titles = ["Model #", "ML Algorithm", "Derivative", "Preprocessing Pipeline", "Preprocessing Strategy", "Oversampler", "Padding Method",
                  "Sensitivity", "Specificity", "Precision", "F1 Score", "Accuracy",
                  "True Positives", "True Negatives", "False Positives", "False Negatives"]
    experiment_data = pd.DataFrame(columns=column_titles)
    for derivative in derivatives:
        for pipeline in pipelines:
            for strategy in strategies:
                try:
                # Download data
                    diagnostic_model.download_data(desired_derivative=derivative, desired_strategy=strategy, desired_pipeline=pipeline, print_stats=False)
                    print()
                    print(f"Downloaded data for parameters -> derivative: {derivative}, pipeline: {pipeline}, strategy: {strategy}")
                except Exception as e:
                    print(f"Error in data downloading: {e}")

                for oversampler in oversamplers:
                    for padder in padding_methods:
                        features, labels = diagnostic_model.features_and_labels(derivative=derivative, pipeline=pipeline, strategy=strategy, filler=padder, print_stats=False, oversampler=oversampler)
                        print()
                        print(f"Extracted features and labels -> filler value: {padder}, oversampler: {oversampler}")

                        for algorithm in algorithms:
                            try:
                                model_performance = diagnostic_model.train_test_fMRI_data_kfold(fMRI_features=features, labels=np.ravel(labels), algorithm=algorithm, k=5, print_stats=False, algorithm_hypertuned=False)

                                model_data = [str(item) for item in [model_count, algorithm, derivative, pipeline, strategy, oversampler, padder] + model_performance[1:5] + [model_performance[0]] + model_performance[5:]]
                                # print(model_data)

                                # Add the model's performance metrics to experiment data
                                experiment_data.loc[len(experiment_data)] = model_data
              
                            except Exception as e:
                                print()
                                print(f"Error in execution of Model-{model_count}: {e}")
                            
                            print()
                            print(f"Model-{model_count}'s testing has been completed!")
                            # print(experiment_data.head())
                            print("-------------------------------------------------------------------------------------------------------------------------------")
                            model_count += 1
      
            print("_______________________________________________________________________________________________________________________________")
    
    return experiment_data


test_data = main_test_multiprocessing()

test_data.to_excel(title="Experiment-2 Test Data Table.xlsx", index=False)

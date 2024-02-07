
import pandas as pd
import diagnostic_model

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

def main_test():
  derivatives = ["rois_aal", "rois_cc200", "rois_cc400", "rois_dosenbach160", "rois_ez", "rois_ho", "rois_tt"]
  algorithms = ["DT", "KNN", "LR", "XGB"]
  pipelines = ["cpac", "niak"]
  strategies = ["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"]

  model_count = 1
  column_titles = ["Model #", "ML Algorithm", "Derivative", "Preprocessing Pipeline", "Preprocessing Strategy",
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

        for algorithm in algorithms:
          try:
              # Run the model; download data if needed, collect features and labels, train and test algorithm, collect performance metrics
              model_performance = diagnostic_model.test_diagnostic_model(derivative=derivative, strategy=strategy, pipeline=pipeline, algorithm=algorithm, print_stats=False, kFold=True)
              
              model_data = [model_count, algorithm, derivative, pipeline, strategy] + model_performance[1:4] + model_performance[0] + model_performance[5:8]
              
              # Add the model's performance metrics to experiment data
              experiment_data = experiment_data.append(pd.Series(model_data, index=column_titles), ignore_index=True)
          
          except Exception as e:
              print(f"Error in execution of Model-{model_count}: {e}")

          print("Model-{model_count}'s testing has been completed!")
          print()
          model_count += 1

  return experiment_data

test_data = main_test()
print(test_data.head())
test_data.to_excel(title="Experiment-2 Test Data Tablexlsx", index=False)

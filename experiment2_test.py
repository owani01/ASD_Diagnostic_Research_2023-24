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
  padding_methods = ["zero", "mean", "median"]

  model_count = 1
  column_titles = ["Model #", "ML Algorithm", "Derivative", "Preprocessing Pipeline", "Preprocessing Strategy", "Padding Method",
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
          for pad in padding_methods:
            try:
              # Run the model; download data if needed, collect features and labels, train and test algorithm, collect performance metrics
              model_performance = diagnostic_model.test_diagnostic_model(derivative=derivative, strategy=strategy, pipeline=pipeline, algorithm=algorithm, filler_value=pad, print_stats=False, kFold=True)
              
              model_data = [str(item) for item in [model_count, algorithm, derivative, pipeline, strategy, pad] + model_performance[1:5] + [model_performance[0]] + model_performance[5:]]
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

test_data = main_test()

test_data.to_excel(title="Experiment-2 Test Data Table.xlsx", index=False)

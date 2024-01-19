import pandas as pd
import nbimporter
import model

derivatives = ["rois_aal", "rois_cc200", "rois_cc400", "rois_dosenbach160", "rois_ez", "rois_ho", "rois_tt"]
algorithms = ["SVM", "RF", "DT", "KNN", "LR", "NB"]
pipelines = ["cpac", "niak"]
strategies = ["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"]

model_count = 1
column_titles = ["Model #", "ML Algorithm", "Derivative", "Preprocessing Pipeline", "Preprocessing Strategy",
                "Sensitivity", "Specificity", "Precision", "F1 Score", "Accuracy",
                "True Positives", "True Negatives", "False Positives", "False Negatives"]
experiment_data = pd.DataFrame(columns=column_titles)

for derivative in derivatives:
    for algorithm in algorithms:
        for pipeline in pipelines:
            for strategy in strategies:
                # Run the model; download data if needed, collect features and labels, train and test algorithm, collect performance metrics
                try:
                    model_performance = model.test_diagnostic_model(derivative=derivative, strategy=strategy, pipeline=pipeline, algorithm=algorithm, print_stats=False)
                except Exception as e:
                    print(f"Error in model execution: {e}")

                # Add the model's performance metrics to experiment data
                model_data = [model_count, algorithm, derivative, pipeline, strategy] + model_performance[1:4] + model_performance[0] + model_performance[5:8]
                experiment_data = experiment_data.append(pd.Series(model_data, index=column_titles), ignore_index=True)

                print("Model-{model_count}'s testing has been completed!")
                model_count += 1

print(experiment_data.head())
experiment_data.to_excel(title="Experiment-2 Test Data Table", index=False)
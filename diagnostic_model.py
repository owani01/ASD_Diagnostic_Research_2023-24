# Import the script function that downloads preprocessed fMRI data
from download_abide_preprocessed_dataset import collect_and_download

import os
import numpy as np
import pandas as pd

# Part of scikit-learn library; Support Vector Classification (SVC), a type of SVM used for classification tasks
from sklearn.svm import SVC

# Part of scikit-learn library; Random Forest Classifier, a type of Random Forest used for classification tasks
from sklearn.ensemble import RandomForestClassifier

# Part of scikit-learn library; Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression

# Part of scikit-learn library; Decision Tree Classifier, a type of Decision Tree used for classification tasks
from sklearn.tree import DecisionTreeClassifier

# Part of scikit-learn library; Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB

# Part of scikit-learn library; K-Neighbors Classifier; a type of K-Nearest Neighbors used for classification tasks
from sklearn.neighbors import KNeighborsClassifier

# XGBoost Classifier Algorithm; a type of gradient-boosted decision trees designed for better speed and performance, used for classification
import xgboost as xgb

# GridSearchCV functionality from scikit-learn; used to hypertune parameters for machine learning model
from sklearn.model_selection import GridSearchCV

# Function from scikit-learn; splits datasets into training and testing sets
from sklearn.model_selection import train_test_split

# Function from scikit-learn; computes parameters relating to the model's performance
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer

# Functionalities from scikit-learn for performing k-fold cross validation of data
from sklearn.model_selection import KFold, cross_val_predict

# Functionalities from scikit-learn for scaling
from sklearn.preprocessing import RobustScaler

# Oversampling methods from imbalanced-learn
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE

def download_data(desired_derivative, desired_strategy, desired_pipeline, print_stats=True):
    # Variables to specify download settings (modify these values as needed)
    download_asd_dir = 'abide_preprocessed_dataset/ASD'  # Path to local folder to download files to for ASD data
    download_tdc_dir = 'abide_preprocessed_dataset/TDC' # Path to local folder to download files to for controls
    desired_diagnosis_asd = 'asd'  # 'asd', 'tdc', or 'both' corresponding to the diagnosis of the participants for whom data should be downloaded
    desired_diagnosis_tdc = 'tdc'  # 'asd', 'tdc', or 'both' corresponding to the diagnosis of the participants for whom data should be downloaded

    # Call the collect and download routine
    collect_and_download(desired_derivative, desired_pipeline, desired_strategy, download_asd_dir, desired_diagnosis_asd, print_stats=print_stats)
    collect_and_download(desired_derivative, desired_pipeline, desired_strategy, download_tdc_dir, desired_diagnosis_tdc, print_stats=print_stats)


# Organizing Features and Labels to input into ML algorithm
def gather_features(data_dir):
    features = []

    for file_name in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, file_name)

        # Check if the file is a timeseries file
        if file_name.endswith('.1D'):
            # Load the timeseries data
            timeseries_data = load_timeseries_data(subject_path)

            # Append features
            features.append(timeseries_data)

    features_list = [np.array(feature) for feature in features]
    return features_list

def load_timeseries_data(file_path):
    # Load timeseries data from the 1D file
    with open(file_path, 'r') as f:
        timeseries = np.loadtxt(f)

    return timeseries

# Define a function for padding time series data
def pad_timeseries_to_length(timeseries, target_length, filler='zero'):
    if len(timeseries) >= target_length:
        return timeseries[:target_length]
    else:
        padding_length = target_length - len(timeseries)
        if filler == 'zero':
            padding = np.zeros((padding_length, timeseries.shape[1]))
        elif filler == 'mean':
            mean_value = np.mean(timeseries, axis=0)
            padding = np.tile(mean_value, (padding_length, 1))
        elif filler == 'median':
            median_value = np.median(timeseries, axis=0)
            padding = np.tile(median_value, (padding_length, 1))
        else:
            raise ValueError("Invalid filler type. Choose from 'zero', 'mean', or 'median'.")

        return np.vstack([timeseries, padding])

def features_and_labels(pipeline, derivative, strategy, filler='zero', print_stats=True, oversampler='None'):
    # Specify the path to the directory containing the downloaded preprocessed data for ASD data
    download_asd_dir = f'abide_preprocessed_dataset/ASD/Outputs/{pipeline}/{strategy}/{derivative}'
    # Make features and labels arrays for ASD data
    features_asd = gather_features(download_asd_dir)
    labels_asd = np.ones(len(features_asd)) # Label 1 for ASD, 0 for TDC

    # Specify the path to the directory containing the downloaded preprocessed data for TDC data
    download_tdc_dir = f'abide_preprocessed_dataset/TDC/Outputs/{pipeline}/{strategy}/{derivative}'
    # Make features and labels arrays for TDC data
    features_tdc = gather_features(download_tdc_dir)
    labels_tdc = np.zeros(len(features_tdc)) # Label 1 for ASD, 0 for TDC

    # Concatenate ASD and TDC features
    all_features = []
    all_features.extend(features_asd)
    all_features.extend(features_tdc)

    # Padding timeseries and # of features for features
    max_length = max(len(ts) for ts in all_features) # Find the maximum length among all timeseries
    all_features_padded = [pad_timeseries_to_length(ts, max_length, filler=filler) for ts in all_features] # Pad each time series to the maximum length
    all_features = np.array(all_features_padded) # Convert the list of padded arrays to a 2D NumPy array
    all_features = all_features.reshape((len(all_features), -1)) # Flatten each time series in all_features to make it a 2D array that is readable for scikit-learn's ML algorithms

    # Concatenate ASD and TDC labels
    all_labels = np.zeros(len(labels_asd) + len(labels_tdc)) # Create an array of zeros with the total length
    all_labels[:len(labels_asd)] = labels_asd # Assign the ASD labels to the first part of the array
    all_labels[len(labels_asd):] = labels_tdc # Assign the TDC labels to the remaining part of the array
    all_labels = np.array(all_labels)

    # Create samplers based on the type of oversampler
    if oversampler == 'Random':
        sampler = RandomOverSampler(random_state=42)
    elif oversampler == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif oversampler == 'ADASYN':
        sampler = ADASYN(sampling_strategy='auto', random_state=42)
    elif oversampler == 'BorderlineSMOTE':
        sampler = BorderlineSMOTE(random_state=42)
    elif oversampler == 'SVMSMOTE':
        sampler = SVMSMOTE(random_state=42)
    else:
        sampler = None

    # Make more samples for the minority class
    if sampler is not None:
        try:
            all_features_resampled, all_labels_resampled = sampler.fit_resample(all_features, all_labels)
        except ValueError as e:
            print("No additional cases were created by the oversampler.") if print_stats else None
            all_features_resampled, all_labels_resampled = all_features, all_labels
        
        all_features, all_labels = all_features_resampled, all_labels_resampled

    if print_stats:
        print("Number of rows in all_features:", len(all_features))
        print("Number of rows in all_labels:", len(all_labels))
        print("Shape of all_features: ", all_features.shape)
        print("Shape of all_labels: ", all_labels.shape)

    features_df, labels_df = pd.DataFrame(all_features), pd.DataFrame(all_labels)

    return features_df, labels_df


# TRAINING AND TESTING ML MODEL (various algorithms)

# Function for hypertuning parameters for XGBoost classifier algorithm
def xgb_classifier_hypertuned(X_train, y_train, print_stats=True):
    parameter_grid = {
        'max depth': [3, 5, 7, 10],
        'learning rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'n_estimators': [50, 100, 200],
        'gamma': [0, 0.1, 0.2]
    }
    if print_stats:
      print("Hyperparameters for XGB model have been defined!")

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss") # Create XGBoost classifier
    if print_stats:
       print("Instance of XGBoost classifier created!")

    # Define a custom scoring function
    def custom_sensitivity_specificity_scoring(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        mean_sensitivity_specificity = (sensitivity + specificity) / 2
        return mean_sensitivity_specificity

    custom_scorer = make_scorer(custom_sensitivity_specificity_scoring) # Create the custom scorer
    if print_stats:
       print("Custom scoring metric created for hyperparameter testing!")

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=parameter_grid, cv=3, scoring=custom_scorer) # Use GridSearchCV for hyperparameter tuning
    if print_stats:
       print("Hyperparameter tuning is complete!")

    grid_search.fit(X_train, y_train)
    if print_stats:
       print("Features have been fitted to hyperparameter tuning algorithm!")

    best_params = grid_search.best_params_ # Get the best hyperparameters
    if print_stats:
       print("Best hyperparameters have been found!")

    best_model = xgb.XGBClassifier(**best_params)
    return best_model

# Function for training/testing ML model (for all algorithms)
def train_test_model(X_train, X_test, y_train, algorithm, print_stats=True, algorithm_hypertuned=False):
    if print_stats:
      print("Entered function for model fitting.")
    
    if algorithm == 'SVM':
        model = SVC()
    elif algorithm == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == 'LR':
        model = LogisticRegression(max_iter=5000)
    elif algorithm == 'DT':
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm == 'NB':
        model = GaussianNB()
    elif algorithm == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    elif algorithm == 'XGB':
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="gpu_hist")
      
    if print_stats:
      print(f"An instance of the {algorithm} algorithm has been created!")

    model.fit(X_train, y_train) # Training the model
    if print_stats:
      print("Model training complete!")

    y_predictions = model.predict(X_test) # Testing the model; getting predictions from the X_test values
    if print_stats:
      print("Model testing complete!")

    return y_predictions

# Function for calculating performance metrics
def calculate_metrics(y_test, y_predictions):
    accuracy = accuracy_score(y_test, y_predictions) # Compute accuracy
    conf_matrix = confusion_matrix(y_test, y_predictions) # Compute confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel() # Extract true positives, false positives, true negatives, and false negatives from confusion matrix
    sensitivity = tp / (tp + fn) # Compute sensitivity (recall)
    specificity = tn / (tn + fp) # Compute specificity
    precision = tp / (tp + fp) # Compute precision
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) # Compute F1 score

    return accuracy, conf_matrix, sensitivity, specificity, precision, f1_score

# Function for training/testing fMRI data with ML model without k-fold cross variation
def train_test_fMRI_data_basic(fMRI_features, labels, algorithm, test_size=0.2, print_stats=True, algorithm_hypertuned=False):
    X_train, X_test, y_train, y_test = train_test_split(fMRI_features, labels, test_size=test_size, random_state=42)
    if print_stats:
      print("Features and labels have been split into training and testing datasets!")

    # Use RobustScaler to normalize the data
    scaler = RobustScaler()
    scaler.fit(X_train) # Fit the scaler to the data
    X_train_scaled = scaler.transform(X_train) # Transform the training data
    X_test_scaled = scaler.transform(X_test) # Transform the test/validation data using the same scaler
    X_train, X_test = X_train_scaled, X_test_scaled

    predictions = train_test_model(X_train, X_test, y_train, algorithm, print_stats=print_stats, algorithm_hypertuned=algorithm_hypertuned)
    if print_stats:
      print("The model has been trained and tested!")

    accuracy, conf_matrix, sensitivity, specificity, precision, f1_score, = calculate_metrics(y_test, predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    if print_stats:
      print("Performance metrics for the model have been calculated!")

    if print_stats:
        print("Accuracy: ", accuracy)
        print("Confusion Matrix: \n", conf_matrix)
        print("Sensitivity: ", sensitivity)
        print("Specificity: ", specificity)
        print("Precision: ", precision)
        print("F1 Score: ", f1_score)

    return [accuracy, conf_matrix, sensitivity, specificity, precision, f1_score, tp, tn, fp, fn]

# Function for training/testing fMRI data with ML model using k-fold cross variation
def train_test_fMRI_data_kfold(fMRI_features, labels, algorithm, k, print_stats=True, algorithm_hypertuned=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    if print_stats:
      print("Using kFold cross validation to random split data into sections is done!")


    total_accuracy = 0
    total_conf_matrix = np.zeros((2, 2))
    total_sensitivity = 0
    total_specificity = 0
    total_precision = 0
    total_f1_score = 0
    splitted_set = 1

    for train_index, test_index in kf.split(fMRI_features, labels):
        if print_stats:
          print(f"kFold set #{splitted_set}")

        X_train, X_test = fMRI_features.iloc[train_index], fMRI_features.iloc[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if print_stats:
          print("Features and labels have been split into training and testing datasets!")

        predictions = train_test_model(X_train, X_test, y_train, algorithm, algorithm_hypertuned=algorithm_hypertuned, print_stats=print_stats) # Train and test model using corresponding algorithm and get predictions
        if print_stats:
          print("Model has been trained and tested!")

        accuracy, conf_matrix, sensitivity, specificity, precision, f1_score = calculate_metrics(y_test, predictions) # Calculate performance metrics
        if print_stats:
          print("Performance metrics for the model have been calculated!")

        total_accuracy += accuracy
        total_conf_matrix += conf_matrix
        total_sensitivity += sensitivity
        total_specificity += specificity
        total_precision += precision
        total_f1_score += f1_score
        splitted_set += 1

    average_accuracy = total_accuracy / k
    average_sensitivity = total_sensitivity / k
    average_specificity = total_specificity / k
    average_precision = total_precision / k
    average_f1_score = total_f1_score / k

    total_tn, total_fp, total_fn, total_tp = total_conf_matrix.ravel()

    if print_stats:
        print("Average Accuracy: ", average_accuracy)
        print("Total Confusion Matrix: \n", total_conf_matrix)
        print("Average Sensitivity: ", average_sensitivity)
        print("Average Specificity: ", average_specificity)
        print("Average Precision: ", average_precision)
        print("Average F1 Score: ", average_f1_score)

    return [average_accuracy, average_sensitivity, average_specificity, average_precision, average_f1_score, total_tp, total_tn, total_fp, total_fn]


def test_diagnostic_model(derivative, strategy, pipeline, algorithm, kFold=True, k=5, test_size=0.2, print_stats=True, filler_value='zero', algorithm_hypertuned=False, oversampler=None):
    download_data(desired_derivative=derivative, desired_strategy=strategy, desired_pipeline=pipeline, print_stats=print_stats)

    features, labels = features_and_labels(derivative=derivative, pipeline=pipeline, strategy=strategy, filler=filler_value, print_stats=print_stats, oversampler=oversampler)

    if kFold:
        return train_test_fMRI_data_kfold(fMRI_features=features, labels=labels, algorithm=algorithm, k=k, print_stats=print_stats, algorithm_hypertuned=algorithm_hypertuned)
    else:
        return train_test_fMRI_data_basic(fMRI_features=features, labels=labels, algorithm=algorithm, test_size=test_size, print_stats=print_stats, algorithm_hypertuned=algorithm_hypertuned)


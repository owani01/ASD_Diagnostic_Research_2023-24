# Import the script function that downloads preprocessed fMRI data
from download_abide_preprocessed_dataset import collect_and_download

import os
import numpy as np

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
def pad_timeseries_to_length(timeseries, target_length):
    if len(timeseries) >= target_length:
        return timeseries[:target_length]
    else:
        padding = np.zeros((target_length - len(timeseries), timeseries.shape[1]))
        return np.vstack([timeseries, padding])

def features_and_labels(pipeline, derivative, strategy, print_stats=True):
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
    all_features_padded = [pad_timeseries_to_length(ts, max_length) for ts in all_features] # Pad each time series to the maximum length
    all_features = np.array(all_features_padded) # Convert the list of padded arrays to a 2D NumPy array
    all_features = all_features.reshape((len(all_features), -1)) # Flatten each time series in all_features to make it a 2D array that is readable for scikit-learn's ML algorithms

    # Concatenate ASD and TDC labels
    all_labels = np.zeros(len(labels_asd) + len(labels_tdc)) # Create an array of zeros with the total length
    all_labels[:len(labels_asd)] = labels_asd # Assign the ASD labels to the first part of the array
    all_labels[len(labels_asd):] = labels_tdc # Assign the TDC labels to the remaining part of the array
    all_labels = np.array(all_labels)

    if print_stats:
        print("Number of rows in all_features:", len(all_features))
        print("Number of rows in all_labels:", len(all_labels))
        print("Shape of all_features: ", all_features.shape)
        print("Shape of all_labels: ", all_labels.shape)

    return all_features, all_labels


# TRAINING AND TESTING ML MODEL (various algorithms)

# Function for hypertuning parameters for XGBoost classifier algorithm
def xgb_classifier_hypertuned(X_train, y_train):
    parameter_grid = {
        'max depth': [3, 5, 7, 10],
        'learning rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'n_estimators': [50, 100, 200],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss") # Create XGBoost classifier

    # Define a custom scoring function
    def custom_sensitivity_specificity_scoring(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        mean_sensitivity_specificity = (sensitivity + specificity) / 2
        return mean_sensitivity_specificity
    
    custom_scorer = make_scorer(custom_sensitivity_specificity_scoring) # Create the custom scorer
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=parameter_grid, cv=3, scoring=custom_scorer) # Use GridSearchCV for hyperparameter tuning
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_ # Get the best hyperparameters

    best_model = xgb.XGBClassifier(**best_params) # Train the model with the best hyperparameters
    
    return best_model

# Function for training/testing ML model (for all algorithms)
def train_test_model(X_train, X_test, y_train, algorithm):
    match_algorithm = {
        'SVM': SVC(),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'LR': LogisticRegression(),
        'DT': DecisionTreeClassifier(random_state=42),
        'NB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGB': xgb_classifier_hypertuned(X_train, y_train)
            # NOTE: Processing time is too high, causing program to crash; MUST tune hyperparameters
    } # This maps a string input to its corresponding class in scikit-learn  

    model = match_algorithm.get(algorithm) # Creating the actual model based on the algorithm parameter 

    model.fit(X_train, y_train) # Training the model

    y_predictions = model.predict(X_test) # Testing the model; getting predictions from the X_test values

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
def train_test_fMRI_data_basic(fMRI_features, labels, algorithm, test_size=0.2, print_stats=True):
    X_train, X_test, y_train, y_test = train_test_split(fMRI_features, labels, test_size=test_size, random_state=42)

    predictions = train_test_model(X_train, X_test, y_train, algorithm)
    
    accuracy, conf_matrix, sensitivity, specificity, precision, f1_score, = calculate_metrics(y_test, predictions)
    tn, fp, fn, tp = conf_matrix.ravel()

    if print_stats:
        print("Average Accuracy: ", accuracy)
        print("Overall Confusion Matrix: \n", conf_matrix)
        print("Average Sensitivity: ", sensitivity)
        print("Average Specificity: ", specificity)
        print("Average Precision: ", precision)
        print("Average F1 Score: ", f1_score)

    return [accuracy, conf_matrix, sensitivity, specificity, precision, f1_score, tp, tn, fp, fn]

# Function for training/testing fMRI data with ML model using k-fold cross variation
def train_test_fMRI_data_kfold(fMRI_features, labels, algorithm, k, print_stats=True):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    total_accuracy = 0
    total_conf_matrix = np.zeros((2, 2))
    total_sensitivity = 0
    total_specificity = 0
    total_precision = 0
    total_f1_score = 0

    for train_index, test_index in kf.split(fMRI_features):
        X_train, X_test = fMRI_features[train_index], fMRI_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        predictions = train_test_model(X_train, X_test, y_train, algorithm) # Train and test model using corresponding algorithm and get predictions 

        accuracy, conf_matrix, sensitivity, specificity, precision, f1_score = calculate_metrics(y_test, predictions) # Calculate performance metrics

        total_accuracy += accuracy
        total_conf_matrix += conf_matrix
        total_sensitivity += sensitivity
        total_specificity += specificity
        total_precision += precision
        total_f1_score += f1_score

    average_accuracy = total_accuracy / k
    average_sensitivity = total_sensitivity / k
    average_specificity = total_specificity / k
    average_precision = total_precision / k
    average_f1_score = total_f1_score / k

    total_tn, total_fp, total_fn, total_tp = total_conf_matrix.ravel()

    if print_stats:
        print("Average Accuracy: ", average_accuracy)
        print("Overall Confusion Matrix: \n", total_conf_matrix)
        print("Average Sensitivity: ", average_sensitivity)
        print("Average Specificity: ", average_specificity)
        print("Average Precision: ", average_precision)
        print("Average F1 Score: ", average_f1_score)

    return [average_accuracy, average_sensitivity, average_specificity, average_precision, average_f1_score, total_tp, total_tn, total_fp, total_fn]


def test_diagnostic_model(derivative, strategy, pipeline, algorithm, kFold=True, k=5, test_size=0.2, print_stats=True):
    download_data(desired_derivative=derivative, desired_strategy=strategy, desired_pipeline=pipeline, print_stats=print_stats)

    features, labels = features_and_labels(derivative=derivative, pipeline=pipeline, strategy=strategy, print_stats=print_stats)

    if kFold:
        return train_test_fMRI_data_kfold(fMRI_features=features, labels=labels, algorithm=algorithm, k=k, print_stats=print_stats)
    else:
        return train_test_fMRI_data_basic(fMRI_features=features, labels=labels, algorithm=algorithm, test_size=test_size, print_stats=print_stats)
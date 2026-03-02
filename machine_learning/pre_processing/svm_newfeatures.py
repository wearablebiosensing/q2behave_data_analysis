import pandas as pd 
import numpy as np 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from jupyter_dash import JupyterDash
from scipy import signal
from scipy.fft import fftshift
import plotly.express as px
from sklearn.decomposition import FactorAnalysis
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import plotly.graph_objects as go

# import dash_core_components as dcc
# import dash_html_components as html
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def svm_cross_validation_with_cm(dataframe, test_size=0.2):
    """
    Trains an SVM model using cross-validation on a given dataframe and returns the confusion matrix for the best model.
    
    Parameters:
    - dataframe (pd.DataFrame): The input data.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed.
    
    Returns:
    - dict: Contains results of the cross-validation and confusion matrix.
    """
    # Split data into train and test sets with shuffling
    train_data, test_data = train_test_split(dataframe, test_size=test_size, shuffle=True,random_state=8) #  random_state=random_state
   
    # Extract features and target variable from training data
    X_train = train_data.drop(columns=['type'])
    y_train = train_data['type']
    
    # Extract features and target variable from test data
    X_test = test_data.drop(columns=['type'])
    y_test = test_data['type']

    # Define a pipeline: Standardization + SVM
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    # Hyperparameters for SVM
    param_grid = {
        'svm__C': [0.1, 1,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80,90,100],
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svm__gamma': ['scale']
    }

    results = []

    # Determine the maximum number of splits based on the class with the fewest samples
    max_splits = min(y_train.value_counts())
    print("max_splits: ",max_splits)
    # Hyperparameter tuning and cross-validation
    for folds in [5, 6,7,8,9, min(10, 20),min(15, 20)]:
        skf = StratifiedKFold(n_splits=folds, shuffle=True)

        grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        train_accuracies, test_accuracies, sensitivities, specificities = [], [], [], []
        for train_index, val_index in skf.split(X_train, y_train):
            X_train_fold, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            best_model.fit(X_train_fold, y_train_fold)
            # Compute confusion matrix for this fold
            cm = confusion_matrix(y_val, best_model.predict(X_val))
            TP = cm[1,1]
            TN = cm[0,0]
            FP = cm[0,1]
            FN = cm[1,0]

            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)

            train_accuracies.append(accuracy_score(y_train_fold, best_model.predict(X_train_fold)))
            test_accuracies.append(accuracy_score(y_val, best_model.predict(X_val)))
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        results.append({
            'folds': folds,
            'best_params': best_params,
            'mean_train_accuracy': np.mean(train_accuracies),
            'mean_test_accuracy': np.mean(test_accuracies),
            'mean_sensitivity': np.mean(sensitivities),
            'mean_specificity': np.mean(specificities),
            'train_test_diff': np.mean(train_accuracies)-np.mean(test_accuracies),
            'spec_sen_diff':  np.abs(np.mean(sensitivities) - np.mean(specificities)),
                
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Identify the model with the highest mean test accuracy
    best_model_params = results_df.sort_values(by="mean_test_accuracy", ascending=False).iloc[0]['best_params']

    # Train the best model on the entire training set
    best_svm = SVC(C=best_model_params['svm__C'], kernel=best_model_params['svm__kernel'], gamma=best_model_params['svm__gamma'])
    best_svm.fit(StandardScaler().fit_transform(X_train), y_train)

    # Predict on the test set and compute the confusion matrix.
    y_pred = best_svm.predict(StandardScaler().fit_transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix
    class_names = y_train.unique()
    print("class_names: ",class_names)
    plot_confusion_matrix(cm, classes=class_names)
    

    return {
        "cross_val_results": results_df,
        "confusion_matrix": cm
    }
############################################################################################################################################
# Features we think are good based on intusion.
############################################################################################################################################
df = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/analysis_table_revised.csv")
print("Number of Participants: ",len(df["Participent"].unique()))
print("Number of Features: ",len(df.columns))
ft_ = ['Accel_X_sum', 'Accel_Y_sum', 'Accel_Z_sum',
       'Gyro_X_sum', 'Gyro_Y_sum', 'Gyro_Z_sum', 
       'Magno_X_sum', 'Magno_Y_sum', 'Magno_Z_sum', ]#'Accel_Vecdis_sum', 'Gyro_Vecdis_sum','Magno_Vecdis_sum',

       # 'Accel_X_zero_cross_rate', 'Accel_Y_zero_cross_rate', 'Accel_Z_zero_cross_rate',
       # 'Gyro_X_zero_cross_rate', 'Gyro_Y_zero_cross_rate', 'Gyro_Z_zero_cross_rate', 
       # 'Magno_X_zero_cross_rate','Magno_Y_zero_cross_rate', 'Magno_Z_zero_cross_rate',

# Subset data for selected features.
subset_df_my_list = df[ft_]
subset_df_my_list["type"] = df["type"] 

# Can do it since the number of rows remanin same only columns change.

# Run the function on the provided data
svm_results_my = svm_cross_validation_with_cm(subset_df_my_list)
print(svm_results_my["cross_val_results"])
import pandas as pd 
import numpy as np 
import os
import sys
import itertools
import pickle
import textwrap
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns
from jupyter_dash import JupyterDash
import seaborn as snsH
import plotly.express as px
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy import stats
from scipy import signal
from scipy.fft import fftshift

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import FactorAnalysis
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut,ShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc #permutation_importance
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import zero_one_loss
from mlxtend.evaluate import bias_variance_decomp



import neptune
import neptune.integrations.sklearn as npt_utils
from neptune.types import File


from collections import Counter
from statistics import mode

from mlxtend.evaluate import bias_variance_decomp
from tensorflow import stack
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from keras.layers import Conv1D, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
sys.path.append('/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/GitQ2Bheave/q2behave_data_analysis/jupyternotebooks/ML_Toolbox/')
from utils_ml import *

########################################################################
# Codes for Behavior Types
# Moving chair =1
# Standing = 2
# Manipulating = 3
# Twirling = 4
# Drumming = 5
# Finger = 6
# NoHyp = 7 is also no hyperactive behavior
# pd.isna(value) =0 is also no hyperactive behavior
# None Type =  -1 
########################################################################
def map_to_strings(value):
    if value == 1:
        return "MC"
    elif value == 2:
        return "SW"
    elif value == 3:
        return "MO"
    elif value == 4:
        return "TH"
    elif value == 5:
        return "DF"
    elif value == 6:
        return "FT"
    elif value == 7:
        return "NoHyp"
    elif value == 0:
        return "NoHyp"

        
# Define a function to apply to each value in the 'Behaviour code' column
def label_behaviour(value):
    if value == 1:
        return 1
    elif value == 2:
        return 1
    elif value == 3:
        return 1
    elif value == 4:
        return 1
    elif value == 5:
        return 1
    elif value == 6:
        return 1
    elif value == 7:
        return 0
    elif value == 0:
        return 0
    else:
        return 0
#### For preprocessing dataset.
def data_clean(df_activity_windowed_psd):
    # 1. Remove the -1 behavior code for talking to peers.
    df_activity_windowed_psd = df_activity_windowed_psd[df_activity_windowed_psd["BehaviorCode"]!=-1]
    # Add participant ID 
    # df_activity_windowed_psd['participantId'] =  df_activity_windowed_psd['filename'].str.split('_').str[1]
    # Add activity ID.
    df_activity_windowed_psd["activityID"] = df_activity_windowed_psd["filename"].str.split('_').str[3]
    # print(df_activity_windowed_psd["BehaviorCode"].unique())   
    #Create a type column to make all fidgeting behaviors as 1 and all non-fidgeting behaviors as 0.
    df_activity_windowed_psd['type'] = df_activity_windowed_psd['BehaviorCode'].apply(label_behaviour)
    # Create a feature name column to map behavior codes back to string variables for easy processing.
    df_activity_windowed_psd["FeatureName"] = df_activity_windowed_psd['BehaviorCode'].apply(map_to_strings)
    return df_activity_windowed_psd

##### Just a helper to get all the columns consisting of Gryo_Max AND Acc_Max. ####################
def return_max_signal_features(df_activity_windowed_psd):
    # Use list comprehension to filter column names containing the substrings 'Gryo_Max' or 'Acc_Max'
    filtered_columns = [col for col in df_activity_windowed_psd.columns if 'Gryo_Max' in col or 'Acc_Max' in col]
    # Print the filtered column names
    print(filtered_columns)
    return filtered_columns
    
# tag ="Clus" = Clustering Based
# tag ="Rus" = Random undersampling.
def my_train_test_split(dataframe,test_size,tag):
    y = dataframe["type"]
    X = dataframe.drop(columns=['type','filename', 'participantId','activityID',"BehaviorCode","FeatureName","WindowNumber"])
    if tag == "Clus":
        print("my_train_test_split() ================================","Clustering Centroids")
        cc = ClusterCentroids(random_state=0)
        X_resampled, y_resampled = cc.fit_resample(X, y)
    elif tag == "Rus":
        print("my_train_test_split() ================================","Random Undersampling")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled  = rus.fit_resample(X, y)
    
    print("Resampled Number of values in each class: ",sorted(Counter(y_resampled).items()))
    X_train_info, X_test_info, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, stratify=y_resampled)
    X_train =X_train_info #X_train_info.drop(columns=['filename', 'participantId','activityID',"BehaviorCode","FeatureName","WindowNumber"])
    X_train = X_resampled
    y_train = y_resampled
    X_test = X_test_info #X_test_info.drop(columns=['filename', 'participantId','activityID',"BehaviorCode","FeatureName","WindowNumber"])
    
    test_data = pd.concat([X_test_info, y_test], axis=1)
    # print("test_data: ",test_data.head())
    print("my_train_test_split()/: X_train: ",X_train.shape, "y_train:",y_train.shape,"Value counts in train:",type(y_train),y_train.value_counts())
    print("my_train_test_split()/: X_test: ", X_test.shape,"y_test:  ", y_test.shape,"Value counts in test:",type(y_test),y_test.value_counts()) 
    resampled_data = pd.concat([ X_resampled, y_resampled], axis=1)
   
    return X_train, y_train,X_test, y_test ,test_data,resampled_data
# Takes in the best SVM model and the training data.
# Returns and calculated performance metrics.
def train_test_accuracies(X_train, y_train,skf,best_model,best_params):
    train_accuracies, test_accuracies, sensitivities, specificities = [], [], [], []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        print("X_train_fold: ",X_train_fold.shape, "y_train_fold: ",y_train_fold.shape)
        best_model.fit(X_train_fold, y_train_fold)
        # Compute confusion matrix for this fold
        cm = confusion_matrix(y_val, best_model.predict(X_val)) # only for calculating speccificity and sensitivity.
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        print("TP: ",TP,"TN: ",TN,"FP: ",FP,"FN: ",FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print("sensitivity: ",sensitivity,"specificity: ",specificity)
        train_accuracies.append(accuracy_score(y_train_fold, best_model.predict(X_train_fold)))
        test_accuracies.append(accuracy_score(y_val, best_model.predict(X_val)))
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    return train_accuracies, test_accuracies, sensitivities, specificities

    
def feature_importance(best_svm, X_train, y_train):
    # Calculate permutation importance
    perm_importance = permutation_importance(best_svm, X_train, y_train, n_repeats=30, random_state=42)
    # Get feature names
    feature_names = X_train.columns
    # Plotting feature importances
    sorted_idx = perm_importance.importances_mean.argsort()
    # df = pd.concat([feature_names[sorted_idx],perm_importance.importances_mean[sorted_idx]],axis=1)
    # Concatenate along rows (stacking one array on top of the other)
    # Concatenate along columns
    concatenated_df = pd.DataFrame({'feature': feature_names[sorted_idx], 'perm_importance': perm_importance.importances_mean[sorted_idx]}, columns=['feature', 'perm_importance'])

    # concatenated_df = pd.DataFrame(np.concatenate([, ], axis=1)).reindex()
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance in SVM Model")
    return plt,concatenated_df
    
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, cl,
               asses, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="orange" if cm[i, j] > thresh else "orange")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def bagging_tree(dataframe, test_size,pid,tag):
#     # tag ="Clus" = Clustering Based
# # tag ="Rus" = Random undersampling.
# def my_train_test_split(dataframe,test_size,tag)
    X_train, y_train,X_test, y_test ,test_data,resampled_data = my_train_test_split(dataframe,test_size=test_size,tag=tag)
    # fit scaler on training data
    X_train_norm = MinMaxScaler().fit(X_train)
    #transform training data
    X_train = pd.DataFrame(X_train_norm.transform(X_train), columns=X_train.columns, index=X_train.index)
    # Calculate the mean of each column
    X_train_means = X_train.mean()
    # Calculate the standard deviation of each column
    X_train_std_devs = X_train.std()
    # print("rf_cross_validation_with_cm():/ Normalized X_Train: \n",X_train.head())
    # print("rf_cross_validation_with_cm():/ MEAN,STD of training data: ",X_train_means,X_train_std_devs)
    # print("rf_cross_validation_with_cm():/ after normalization X_train type:", type(X_train), len(X_train))
    X_test_norm = MinMaxScaler().fit(X_test)

    # transform testing data
    X_test = pd.DataFrame(X_test_norm.transform(X_test), columns=X_test.columns, index=X_test.index)
    # print("rf_cross_validation_with_cm():/ X_test Value Counts: \n",X_test.value_counts())
    # print("rf_cross_validation_with_cm():/ Normalized X_test: \n",X_test.head())
    

    # Define base estimator (Decision Tree) for Bagging
    base_estimator = RandomForestClassifier()
    
    # Define the Bagging Classifier
    bagging_classifier = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=8)
    
    # Define the pipeline
    pipeline = Pipeline([
        ('bagging', bagging_classifier)
    ])

    # Hyperparameters for Bagging Classifier
    param_grid = {
        'bagging__base_estimator__max_depth': [None, 10, 20, 30],
        'bagging__max_samples': [0.5, 0.7, 0.9],
        'bagging__max_features': [0.5, 0.7, 0.9]
    }


    results = []

    # Determine the maximum number of splits based on the class with the fewest samples
    max_splits = min(y_train.value_counts())

    # Hyperparameter tuning and cross-validation
    for folds in [2,3]:#[5, 6, 7, 8, 9, min(10, max_splits)]:
        run = neptune.init_run(
            project="shehjar/ADHD-SVM",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMGQxM2Q4ZC00ODE4LTRhNDMtYWY1Yy1kMGIzZTI5MzcxMzUifQ==",
            name= str(folds) + "-Fold"+ "test_size_" + str(test_size) + "_RF"   # Set your custom run name here
        )  # your credential
        run["pid"] = str(pid)
        run['feature_list'] = dataframe.columns
        # File
        run["data/feature_table"].upload(File.as_html(dataframe))
        run['original']  = str(dataframe["type"].value_counts())
        run['train_samples_undersampled'] = str(y_train.value_counts())
        run['test_samples_undersampled'] = str(y_test.value_counts())
        run["tag"] = tag

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=8)
        grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_probs = best_model.predict_proba(X_test)
        print("y_pred_probs: \n",y_pred_probs)
        dataset = pd.DataFrame({'Prob_0': y_pred_probs[:, 0], 'Prob_1': y_pred_probs[:, 1]})

        print("dataset pred probs: \n",dataset)
        test_data["RF_predictions"] = y_pred
        test_data["Prob_0"] = y_pred_probs[:, 0]
        test_data["Prob_1"] = y_pred_probs[:, 1]
        ## Probability plots ###
        run["data/test_data"].upload(File.as_html(test_data))
        
        ###### Returns a scikit-learn classifier summary that includes ##################
        # All classifier parameters
        # Pickled estimator (model)
        # Test predictions
        # Test predictions probabilities
        # Test scores
        # Model performance visualizations
        #################################################################################
        run["cls"]="RandomForest"
        run["cls_summary"] = npt_utils.create_classifier_summary(best_model, X_train, X_test, y_train, y_test)
        run["params"] = best_params
        train_accuracies, test_accuracies, sensitivities, specificities = train_test_accuracies(X_train, y_train,skf,best_model,best_params)
        
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        best_model, X_train.values, y_train.values, X_test.values, y_test.values, 
        loss='0-1_loss',
        random_seed=123)
        run["loss"] = str(avg_expected_loss)
        run["bias"] = avg_bias
        run["variance"] = avg_var
        zero_one_loss(y_test,y_pred)
        print('Average expected loss--After pruning: %.3f' % avg_expected_loss)
        print('Average bias--After pruning: %.3f' % avg_bias)
        print('Average variance--After pruning: %.3f' % avg_var)
        print('Sklearn 0-1 loss--After pruning: %.3f' % zero_one_loss(y_test,y_pred))
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        ## Feature Importance Plots ###
        plt_features,df_feature_perm = feature_importance(best_model, X_train, y_train)
        run["data/feature_importance"].upload(File.as_html(df_feature_perm))

        file_path_feature_importance = "feature_importance.png"
        plt_features.savefig(file_path_feature_importance)
        run['feature_importance'].upload(neptune.types.File(file_path_feature_importance))
        
        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for the test set predictions')
        plt.legend(loc="lower right")
        # Save the ROC curve as an image
        roc_curve_image_path = 'roc_curve_class.png'
        plt.savefig(roc_curve_image_path)
        # Log the ROC curve image
        run['roc_curve'].upload(neptune.types.File(roc_curve_image_path))
        plt.show()
        
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        print("precision: ",precision,"recall: ",recall,"f1_score",f1_score)
        run['precision'] = np.mean(precision)
        run['recall'] = np.mean(recall)
        run['f1_score'] = np.mean(f1_score)
        run['mean_train_accuracy'] = np.mean(train_accuracies)
        run['mean_test_accuracy'] = np.mean(test_accuracies)
        run['mean_sensitivity'] = np.mean(sensitivities)

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

    return results_df

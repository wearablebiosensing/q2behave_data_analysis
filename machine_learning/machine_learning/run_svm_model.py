import pandas as pd 
import numpy as np 
import os
import sys
import itertools
import pickle
import textwrap

from plotly.subplots import make_subplots
import seaborn as sns
from jupyter_dash import JupyterDash
import seaborn as snsH
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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



import neptune
import neptune.integrations.sklearn as npt_utils
from neptune.types import File


from collections import Counter
from statistics import mode


sys.path.append('/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/GitQ2Bheave/q2behave_data_analysis/jupyternotebooks/signal_processing_box')
sys.path.append('/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/GitQ2Bheave/q2behave_data_analysis/jupyternotebooks/ML_Toolbox/')
from utils_ml import *

from statistical_features import *
from power_spectral_features import *
from filtering_module import *
from helpers import *
from ml_svm import *
# Read the two feature sets
root_feature_folder = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/CodeOutputs/Windowing_Features/"
df_activity_windowed_psd = pd.read_csv(root_feature_folder + "psd_features_two_second.csv")
df_activity_windowed_stat = pd.read_csv(root_feature_folder+ "stat_features_two_second.csv")
df_activity_windowed =  pd.read_csv(root_feature_folder+  "all_features_two_second.csv")

filtered_columns = return_max_signal_features(df_activity_windowed)

df_activity_windowed = data_clean(df_activity_windowed)
# Filtered out behaviors.
df_activity_windowed = df_activity_windowed[df_activity_windowed['FeatureName'].isin(['NoHyp','TH', 'MO', 'FT', 'DF','MC'])] # ,'MC','SW'
print("New BH code:",df_activity_windowed["BehaviorCode"].unique())
def activity_segments(df_activity_windowed):
    ##################################################################################################################################
    df_activity_windowed_psd_all_act = df_activity_windowed[[
    "Filtered_Accel_X_max_power_psd",
    "Filtered_Accel_X_min_power_psd",
    "Filtered_Accel_X_skewness_power_psd",
    "Filtered_Accel_X_kurtosis_power_psd",
    "Filtered_Accel_X_mean_power_psd",
    "Filtered_Accel_X_sum_freq_diff_power_psd",
    "Filtered_Accel_X_average_power_power_psd",
    "Filtered_Accel_Xfeature_num_peaks_power_psd",
    "Filtered_Accel_Y_max_power_psd",
    "Filtered_Accel_Y_min_power_psd",
    "Filtered_Accel_Y_skewness_power_psd",
    "Filtered_Accel_Y_kurtosis_power_psd",
    "Filtered_Accel_Y_mean_power_psd",
    "Filtered_Accel_Y_sum_freq_diff_power_psd",
    "Filtered_Accel_Y_average_power_power_psd",
    "Filtered_Accel_Yfeature_num_peaks_power_psd",
    "Filtered_Accel_Z_max_power_psd",
    "Filtered_Accel_Z_min_power_psd",
    "Filtered_Accel_Z_skewness_power_psd",
    "Filtered_Accel_Z_kurtosis_power_psd",
    "Filtered_Accel_Z_mean_power_psd",
    "Filtered_Accel_Z_sum_freq_diff_power_psd",
    "Filtered_Accel_Z_average_power_power_psd",
    "Filtered_Accel_Zfeature_num_peaks_power_psd",
    "Filtered_Gryo_X_max_power_psd",
    "Filtered_Gryo_X_min_power_psd",
    "Filtered_Gryo_X_skewness_power_psd",
    "Filtered_Gryo_X_kurtosis_power_psd",
    "Filtered_Gryo_X_mean_power_psd",
    "Filtered_Gryo_X_sum_freq_diff_power_psd",
    "Filtered_Gryo_X_average_power_power_psd",
    "Filtered_Gryo_Xfeature_num_peaks_power_psd",
    "Filtered_Gryo_Y_max_power_psd",
    "Filtered_Gryo_Y_min_power_psd",
    "Filtered_Gryo_Y_skewness_power_psd",
    "Filtered_Gryo_Y_kurtosis_power_psd",
    "Filtered_Gryo_Y_mean_power_psd",
    "Filtered_Gryo_Y_sum_freq_diff_power_psd",
    "Filtered_Gryo_Y_average_power_power_psd",
    "Filtered_Gryo_Yfeature_num_peaks_power_psd",
    "Filtered_Gryo_Z_max_power_psd",
    "Filtered_Gryo_Z_min_power_psd",
    "Filtered_Gryo_Z_skewness_power_psd",
    "Filtered_Gryo_Z_kurtosis_power_psd",
    "Filtered_Gryo_Z_mean_power_psd",
    "Filtered_Gryo_Z_sum_freq_diff_power_psd",
    "Filtered_Gryo_Z_average_power_power_psd",
    "Filtered_Gryo_Zfeature_num_peaks_power_psd", 'WindowNumber', 'filename', 'participantId', 'activityID', 'type','BehaviorCode', 'FeatureName']]
    df_activity_windowed_psd_A1= df_activity_windowed_psd_all_act[df_activity_windowed_psd_all_act["activityID"]=="A1.csv"]
    df_activity_windowed_psd_A2= df_activity_windowed_psd_all_act[df_activity_windowed_psd_all_act["activityID"]=="A2.csv"]
    df_activity_windowed_psd_A3= df_activity_windowed_psd_all_act[df_activity_windowed_psd_all_act["activityID"]=="A3.csv"]
    df_activity_windowed_psd_A4= df_activity_windowed_psd_all_act[df_activity_windowed_psd_all_act["activityID"]=="A4.csv"]
    df_activity_windowed_psd_all_act = df_activity_windowed_psd_all_act.dropna()
    df_activity_windowed_psd_A1 = df_activity_windowed_psd_A1.dropna()
    df_activity_windowed_psd_A2 = df_activity_windowed_psd_A2.dropna()
    df_activity_windowed_psd_A3 = df_activity_windowed_psd_A3.dropna()
    df_activity_windowed_psd_A4 = df_activity_windowed_psd_A4.dropna()
    return df_activity_windowed_psd_A1,df_activity_windowed_psd_A2,df_activity_windowed_psd_A3,df_activity_windowed_psd_A4,df_activity_windowed_psd_all_act

df_activity_windowed_psd_A1,df_activity_windowed_psd_A2,df_activity_windowed_psd_A3,df_activity_windowed_psd_A4,df_activity_windowed_psd_all_act = activity_segments(df_activity_windowed)
test_set_size = 0.2

svm_cross_validation_with_cm(df_activity_windowed_psd_all_act, test_size=0.2,tag="Clus")
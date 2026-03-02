import pandas as pd 
import numpy as np 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
# from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from jupyter_dash import JupyterDash
from scipy import signal
from scipy.fft import fftshift
import plotly.graph_objects as go
import matplotlib.dates as mdates
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch
from scipy.signal import find_peaks
import scipy as sc
from scipy.stats import skew, kurtosis
import scipy.fftpack                 
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import chirp, peak_widths
import math



def average_sampling_rate(df):
    sr_dict = df["NewClock"].value_counts().to_dict()
    print("average_sampling_rate():/ sr_dict: ",sr_dict)
    # Step 1: Calculate the sum of all values
    total_sum = sum(sr_dict.values())
    # Step 2: Count the number of values
    num_values = len(sr_dict)
    if num_values !=0:
        print("num_values: ",num_values)
        print("total_sum: ",total_sum)
        # Step 3: Calculate the average
        average_sr = total_sum / num_values
        return average_sr
    else:
        return 0
## Add codes to string behaviour types.##
def pre_datacleaning(row):
    ################################################
    # Filter out specific behavior types.
    ################################################
    if "Standing" in row['HyperactiveBehaviourType']:
        return 1
    elif "chair" in row['HyperactiveBehaviourType']:
        return 2
    elif "Manipulating" in row['HyperactiveBehaviourType']:
        return 3
    elif "Twirling" in row['HyperactiveBehaviourType']:
        return 4
    elif "Drumming" in row['HyperactiveBehaviourType']:
        return 5
    elif "Finger" in row['HyperactiveBehaviourType']:
        return 6
    elif "NoHyp" in row['HyperactiveBehaviourType']:
        return 0
    else:
        return "None"  # Or whatever default behavior you want
# Convert the elapsed_time strings to seconds
def to_seconds(time_string):
    minutes, seconds = map(int, time_string.split(":")[1:])
    return minutes * 60 + seconds

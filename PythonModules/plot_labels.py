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

# import dash_core_components as dcc
# import dash_html_components as html

labels_folder_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/LabelsAll/"
def plot_bar_plot(grouped_data,filename):
     # Plotting the bar plot using matplotlib
    fig, ax = plt.subplots(figsize=(15, 8))
    bars = ax.bar(grouped_data.index, grouped_data.values, color='skyblue')
    # Adding value annotations on top of each bar with increased font size
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), 
                ha='center', va='bottom', fontsize=20)  # Increased fontsize to 14
    
#     plt.bar(grouped_data.index, grouped_data.values, color='skyblue')
    plt.title('Duration for Different Hyperactive Behaviour Types \n' + filename, fontsize=20)
    plt.xlabel('Hyperactive Behaviour Type',fontsize=20)
    plt.ylabel('Duration (in seconds)',fontsize=20)
    plt.xticks(rotation=15)
    plt.yticks(fontsize=18)  # y-axis ticks font size
    plt.tight_layout()
    return plt

# Master dataframe to store data from all CSV files
master_data_frame = pd.DataFrame()
master_data_frame_2 = pd.DataFrame(columns=['nan', 'Drumming fingers ',
       'Manipulating object (turning pen over & over in hand)',
       'Twirling your hair', 'Hyperactive Behaviour Type',
       'Standing up while working ', 'Moving the chair back and forth',
       'Talking to peers', 'Talking to self',
       'Tapping foot/ Bouncing Leg', 'Finger tapping', 'None ',
       ' Bouncing leg / Shaking leg'])
for filename in os.listdir(labels_folder_path):
    if filename.endswith(".csv"):
        filepath = os.path.join(labels_folder_path, filename)
        print("filepath: ",filepath)
        df_labels = pd.read_csv(filepath,header=None)
        df_labels = df_labels.drop(0)
        df_labels.columns = ["Elapsed_Time","OffSeat","Hyperactive_Restless","HyperactiveBehaviourType","SecondHyperactiveBehaviourType","Inattentive","InattentiveBehaviourType","SecondInattentiveBehaviourType"]
        filtered_data = df_labels[df_labels["HyperactiveBehaviourType"] != "None"]
        filtered_data["PID"] = filename.split("_")[1].split(".")[0]
        filtered_data["activity_id"] = filename.split("_")[0]
        #master_data_frame = master_data_frame.append(filtered_data)
        master_data_frame = pd.concat([master_data_frame, filtered_data], ignore_index=True)

        #print("HyperactiveBehaviourType: ",df_labels["HyperactiveBehaviourType"].unique())
        # Group by the "Hyperactive Behaviour Type" and count the occurrences
        grouped_data = filtered_data.groupby("HyperactiveBehaviourType").size()
        #print("grouped_data: ",grouped_data,type(grouped_data))
        #print(grouped_data.to_frame())
        print("=====================================================================================================================")
        plt = plot_bar_plot(grouped_data,filename)
        plt.show()

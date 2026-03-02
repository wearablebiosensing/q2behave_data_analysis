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
########################################################################################################
## Analysis of segmented behaviour types data.
########################################################################################################
# HyperactiveBehaviourType
def get_segments_for_behaviour(data, behaviour_type):
    """
    Returns the start and end indices of contiguous segments for a given behaviour type.
    """
    # Create a mask for rows matching the given behaviour type
    mask = (data['HyperactiveBehaviourType'] == behaviour_type)
    
    # Add sentinel values (False) to the beginning and end to capture boundary segments
    extended_mask = [False] + list(mask) + [False]
    
    # Use a list comprehension to capture the start and end index of each segment
    segments = [(start, end) for start, end in zip(range(len(extended_mask) - 1), range(1, len(extended_mask)))
                if extended_mask[start] != extended_mask[end]]
    
    # Convert pairs of start and end indices into segments
    segments = [(segments[i][0], segments[i+1][0]-1) for i in range(0, len(segments), 2)]
    print("segments: ",segments)
    return segments

# # Getting the segments for the "Finger tapping" behaviour type
# finger_tapping_segments = get_segments_for_behaviour(data, "Finger tapping")
# finger_tapping_segments, len(finger_tapping_segments)
def save_segments_to_csv(data, behaviour_type, segments,output_dir,file):
    """
    Save each segment of data for a given behaviour type to separate CSV files.
    """
    output_files = []

    for idx, (start, end) in enumerate(segments):
        # Extracting the segment from the data
        segment_data = data.iloc[start:end+1]
        print("behaviour_type===== ",behaviour_type)
        pid_info = file.split(".")[0]
        # Generate the filename
        filename = os.path.join(output_dir, pid_info + "_"+f"{behaviour_type.replace(' ', '_').replace('/', '_').replace('&', 'and')}_{idx+1}.csv")
        print("filename: ",filename)
        output_files.append(filename)
        
        # Write the segment to a CSV file
        segment_data.to_csv(filename, index=False)

    return output_files

# Folder where all the synced files are placed. 
root_read_folder = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/RenamedData/"

list_files = os.listdir(root_read_folder)
print("list_files: ",list_files)

for file in list_files:
    print("file: =====",file)
    if file != ".DS_Store":
        df = pd.read_csv(root_read_folder + file,low_memory=False)
        df = df.fillna("NoHyp") # nan which is a float value is required as it is the non hyperactive type behaviour.
        # Behaviour type list and the type of the variable:  nan <class 'float'>
        bh_type_list = df["HyperactiveBehaviourType"].unique()
        print("bh_type_list: ",bh_type_list)
        for i in bh_type_list:
            print("Behaviour type list and the type of the variable: ",i, type(i))
        cleaned_list = [item for item in bh_type_list if item != '-100' and item != -100.0 and  item !='None'] #item != float('nan')
        print("cleaned_list: ",cleaned_list)
        for behaviour_type in cleaned_list:
            # Getting the segments for the "Finger tapping" behaviour type
            finger_tapping_segments = get_segments_for_behaviour(df, behaviour_type)
            # finger_tapping_segments, len(finger_tapping_segments)
            output_dir = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/BlobsOFSegments/"
            # # Saving the segments for the "Finger tapping" behaviour type to separate CSV files
            finger_tapping_files = save_segments_to_csv(df, behaviour_type, finger_tapping_segments,output_dir,file)
            finger_tapping_files


import pandas as pd
import numpy as np
import os 
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
folder_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/Session6_DC_06_19_2023/SO4'  # replace with the path to your folder
filtered_dfs = []
##################################################################################################
# To find missing data for session 6.
##################################################################################################
# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
    full_path = os.path.join(folder_path, csv_file)
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(full_path)
    # print(df.columns)
    print("Data TS: ",df[' DataTS'][0].split(" ")[1]) #  06/15/2022 10:21:25.396 AM
    
    # # Check if "06/19/2023" exists in any row of the ' DataTS' column
    # if df[' DataTS'].str.contains("06/19/2023").any():
    #     print("Found the DATA=========================")
    #     filtered_dfs.append(df)
     
    # Check if "06/19/2023" exists in any row of the ' DataTS' column
    if df[' DataTS'].str.contains("06/19/2023").any():
        print("Found the DATA=========================")
        filtered_dfs.append(df)
print(filtered_dfs)
# Now, filtered_dfs will contain all DataFrames with "06/19/2023" in the "DataTS" column


import pandas as pd
import numpy as np
import os 
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

##########################################################################################
# OUTPUT : /syncedfiles Folder on Desktop computer.
##########################################################################################
# 00:01:37, 00:01:37, 00:01:37,00:01:37, 00:01:37, 00:01:37, 00:01:37, 00:01:37, 00:01:37, 00:01:38, 00:01:38, 00:01:38,00:01:38
# df is the seperated activity based dataframe within the A1 watch folder.
def create_seconds_array(df):
    # Get the start and end timestamp
    star_ts = df["timestamp_str"][0]
    end_ts = df["timestamp_str"][len(df["timestamp_str"])-1]
    # Convert the start and end timestamp in dt format
    start_time = datetime.strptime(star_ts, '%H:%M:%S.%f')
    end_time = datetime.strptime(end_ts, '%H:%M:%S.%f')
    # subtract and calcalte the duration.
    duration = end_time - start_time
    duration_in_seconds = duration.total_seconds()
    duration_in_minutes = duration_in_seconds / 60
    # Define the sampling rate and duration
    sampling_rate = len(df["timestamp_str"])/duration_in_seconds #30  # 30 Hz
    duration = duration_in_seconds #int(duration_in_minutes)* 60  #5 * 60  # 5 minutes in seconds
    # Calculate the number of total samples
    total_samples = int(sampling_rate * duration)

    # Create the array of seconds
    seconds_array = np.linspace(0, duration, total_samples)

    # Convert the seconds to hours, minutes, and seconds
    hours = seconds_array // 3600
    minutes = (seconds_array % 3600) // 60
    seconds = seconds_array % 60
    seconds_array_formatted_str = []
    for i in range(len(seconds_array)):
        seconds_array_formatted_str.append(f"{int(hours[i]):02}:{int(minutes[i]):02}:{int(seconds[i]):02}")
    df["seconds_array"] = seconds_array_formatted_str
    print("Both should be same","TS Array in dataframe",len(df["timestamp_str"]),"seconds_array TS: ",len(seconds_array_formatted_str))
    return df
# Takes in a df with smart watch dataset.
# returns a df by creating a new column NewClock in the original df.
def restart_the_clock(df):
    # get the number of samples in each second.
    ts_value_counts = df["timestamp_str_seconds"].value_counts()
    # convert it onto a dictionary.
    ts_value_counts_dict = ts_value_counts.to_dict()
    avaliable_ts_list = ts_value_counts_dict.keys()
    timestamp_datetime = [datetime.strptime(ts, "%H:%M:%S") for ts in avaliable_ts_list]
    sorted_timestamps = sorted(timestamp_datetime)
    sorted_timestamp_strings = [dt.strftime("%H:%M:%S") for dt in sorted_timestamps]
    # Create a new dictionary with sorted keys i.e ts_dict.
    ts_dict = {}
    for key in sorted_timestamp_strings:
        ts_dict[key] = ts_value_counts_dict[key]
    assert len(ts_value_counts_dict) == len(ts_dict), "Length of the sorted TD dict should be the same as the original TS dict"
    # Initialize the new dictionary
    new_ts_dict = {} # this is the TS dict with the new time clock starting at "00:00:00"
    # Convert the initial timestamp to a datetime object
    current_time = datetime.strptime("00:00:01", "%H:%M:%S")

    # Iterate through the original dictionary
    for timestamp, value in ts_dict.items():
        # Store the value in the new dictionary with the formatted timestamp
        new_ts_dict[current_time.strftime("%H:%M:%S")] = value

        # Increment the current time based on the original timestamp increment
        current_time += timedelta(seconds=1)
    result_list = []
    for key, value in new_ts_dict.items():
        result_list.extend([key] * value)
    assert len(result_list) == df.shape[0], "Length of the new time clock should be same as the length of of the watch df"
    df["NewClock"] = result_list
    return df
# merge the labels df and the watch df.
def merge_df_diff_rows(df1,df2):
    # Assuming df1 and df2 are your dataframes
    # Calculate the difference in the number of rows
    row_difference = len(df1) - len(df2)

    # Create a dataframe with "-100" values to match the number of rows to be added
    additional_rows = pd.DataFrame([[-100] * df2.shape[1]] * row_difference, columns=df2.columns)

    # Concatenate df2 and additional_rows along the rows axis
    df2 = pd.concat([df2, additional_rows], ignore_index=True)

    # Now, df2 will have the same length as df1, and you can proceed to merge them as columns
    merged_df = pd.concat([df1, df2], axis=1)
    return merged_df
    # Now, merged_df will have a shape of (16264, 30)

def upsample_labeled_data(value_counts_dict,p6_labels_subset):
    list_of_all_rows = []
    print("upsample_labeled_data():/ ",p6_labels_subset.columns)
    # Convert the Timestamp column by adding  leading 0 of the labeled dataset.
    p6_labels_subset['new_zeroTS'] = p6_labels_subset['Elapsed_Time'].str.zfill(8)
    for index,row in p6_labels_subset.iterrows():
        # print(row["new_zeroTS"],type(row["new_zeroTS"]))
        if row["new_zeroTS"] in value_counts_dict:
            # Row to be duplicated.
            num_time_dup = value_counts_dict[row["new_zeroTS"]] # Number of times to be duplicated.
            #print("Num times to be reprated",num_time_dup)
            dup_rows = []
            for i in range(num_time_dup):
    #             print("i: ",i)
    #             print("row:",row.tolist())
                dup_rows.append(row.tolist())
            #print("Duplicated rows: ",len(dup_rows))
            dup_rows_df = pd.DataFrame(dup_rows)
    #         print("dup_rows_df: ",dup_rows_df)
           # print("dup_rows_df: ",dup_rows_df.shape)
            list_of_all_rows.append(dup_rows_df)
 #   print(list_of_all_rows)
    result_upsampled_labels_df = pd.concat(list_of_all_rows, axis=0)
    result_upsampled_labels_df.columns = p6_labels_subset.columns.to_list()
    return result_upsampled_labels_df

### Doing this as two participants data is combined in one excel file in the labeling dataset. 
def create_two_data_frames(df_labels_a1):
    ### Putcome 1 dataset per participnt
    # Select the first 10 columns
    p5_labels_subset = df_labels_a1.iloc[:, :9]
    p5_labels_subset.columns = ["Elapsed_Time","OffSeat","Hyperactive_Restless","HyperactiveBehaviourType","SecondHyperactiveBehaviourType","Inattentive","Inattentive","SecondInattentive"]
    # Drop the first row
    p5_labels_subset = p5_labels_subset.drop(0)
#     p6_labels_subset = df_labels_a1.iloc[:, [0] + list(range(-8, 0))]
#     p6_labels_subset.columns = ["Elapsed_Time","OffSeat","Hyperactive_Restless","HyperactiveBehaviourType","SecondHyperactiveBehaviourType","Inattentive","Inattentive","SecondInattentive"]
#     # Drop the first row
#     p6_labels_subset = p6_labels_subset.drop(0)
    return p5_labels_subset #, p6_labels_subset

# INPUT OF THIS CELL IS A1 SMART WATCH DATA FOR ONE PARTICIPANT AND THE LABLED EXCEL FILE FORM GOOGLE DRIVE.
## THIS IS THE MAIN FUNCTION THAT NEEDS TO RUN FOR MULTIPLE CSV FILES.
def run_main_algorithm(df,p6_labels_subset):
    #print("run_main_algorithm: ",df['timestamp_str'][0],type(df['timestamp_str'][0]))
    # Apply the function to the entire column to create a timestamp_str column consisting of ts format "09:52.00".
    df['timestamp_str_seconds'] =  df['timestamp_str'].str.split('.').str[0] #df['timestamp_str'].apply(convert_timestamp)
    # Create the seconds array restarting the clock at "00:00:00"
    df_restarted_clock = restart_the_clock(df)
    # First resample the lables.
    ts_count_dict = df_restarted_clock["NewClock"].value_counts()
    result_upsampled_labels_df = upsample_labeled_data(ts_count_dict,p6_labels_subset)
    # Now merge the new smartwatch data frame with restarted timestamps with the original labeled dataframe. 
    merged_lables = merge_df_diff_rows(df_restarted_clock,result_upsampled_labels_df)
    return merged_lables

def create_results_per_participant(pid,root_session):
    # Iterate over all subfolders in the root folder
    for foldername in os.listdir(root_session):
        folder_path = os.path.join(root_session, foldername)
        print("folder_path: ",folder_path)
        # Check if the item is a directory and starts with 'A'
        if os.path.isdir(folder_path) and foldername.startswith('A'):
            # Initialize a list to store P6 file paths
            p6_file_paths = []
            labels_file_path = None
            # Iterate over the files in the current subfolder
            for filename in os.listdir(folder_path):
                file = os.path.join(folder_path, filename)
               # print("filename: ",filename)
                # Check if the file starts with 'P6' and has a '.csv' extension
                if filename.startswith(pid) and filename.endswith(".csv"):
               #     print("SmartWatch filename: \n ",filename)
                    p6_file_paths.append(file)
                # Check if the file ends with 'Labels.csv'
                elif filename.endswith("_"+pid +".csv"):
                #    print("Labels fileName:  \n ",filename)
                    labels_file_path = file
            print("create_results_per_participant(): / p6_file_paths of PID",pid, p6_file_paths)
            # Check if there are P6 files and a Labels file in the current subfolder
            if p6_file_paths and labels_file_path:
                try:
                    print("Labels Path: \n", labels_file_path)
                    # Read the "Labels" file into a pandas DataFrame
                    labels_df = pd.read_csv(labels_file_path)
                    #print("Label Columns", labels_df.columns)
                    # Call the function to create two data frames
                    df1 = create_two_data_frames(labels_df)
                    # Iterate over the P6 files and process each one
                    for p6_file_path in p6_file_paths:
                        print("P6 File Path: \n", p6_file_path)

                        # Read the current "P6" file into a pandas DataFrame
                        p6_df = pd.read_csv(p6_file_path)

                        # Call the main algorithm function for each P6 file
                        upsampled_result_labels = run_main_algorithm(p6_df, df1)

                        # Get the directory path
                        directory_path = os.path.dirname(p6_file_path)

                        # Create a new result file name
                        new_result_name = "result_" + p6_file_path.split("/")[-1]
                        print("New Result Name: ", new_result_name)
                        print("Directory Path: ", root_session)
                        # Save the result to a new CSV file
                        upsampled_result_labels.to_csv(os.path.join(root_session + "syncedfiles/", new_result_name), index=False)
                        upsampled_result_labels.to_csv(os.path.join(directory_path, new_result_name), index=False)
                except pd.errors.EmptyDataError:
                    # Skip empty CSV files
                    print(f"Skipped empty file: {labels_file_path}")

## READ INPUTS
main_root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/"
SESSION_ID = "Session10_DC_07_30_2023/" # Eg: Session5_DC_06_03_2023/
root_session = main_root + SESSION_ID
pid = "P24"
create_results_per_participant(pid,root_session)

# run_main_algorithm(df,p6_labels_subset)
# Call the main algorithm function for each P6 file
# df_watch_data = pd.read_csv(root_session + "A1/P8_SSW4WearOS03_A1.csv")
# df_labeled_data = pd.read_csv(root_session + "/A4/A4_P6.csv")
# df1 = create_two_data_frames(df_labeled_data)
# upsampled_result_labels = run_main_algorithm(df_watch_data, df1)
# upsampled_result_labels.to_csv(root_session + "A4/result_P6_SSW4WearOS03_A4.csv")
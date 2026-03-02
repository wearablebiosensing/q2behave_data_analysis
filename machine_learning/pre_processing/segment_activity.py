import pandas as pd
import numpy as np
import os 
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
###########################################################################################################################################################################################
# This code creates folders A1,A2,A3,A4 and 
# sorted_ts_SSW4WearOS03.csv with timestamps sorted in acending order.
###########################################################################################################################################################################################
#watchID ="SSW4WearOS03"
def read_all_sub_files_in_watch(watchID):
    watch_one = []
    file_paths = os.listdir(root_path+watchID)
    for file in file_paths:
        if file.endswith(".csv"):
            #print("root_path",root_path)
            #print("watch_one",watch_.one)
            #print("file: ",file)
            read_path = root_path + watchID + "/"+ file
            #print("read_path: ",read_path)
            df = pd.read_csv(read_path)
            watch_one.append(df)
            #print("watch_one",watch_one)
    watch_one_df = pd.concat(watch_one)
    # Concatinate all watch files into one file.
    watch_one_df["WatchID"] = watchID
    print("read_all_sub_files_in_watch(): ", watch_one_df)
    # Convert the string ("04/29/2023 09:45:48.329 AM") formatted TS to date and time.
    watch_one_df['timestamp'] = pd.to_datetime(watch_one_df[" DataTS"], format=' %m/%d/%Y %I:%M:%S.%f %p')
    # sort values by the converted timestamps in acending order.
    watch_one_df = watch_one_df.sort_values(by='timestamp')
    # reindex the pandas df to get the correct index subsets.
    watch_one_df = watch_one_df.reset_index()
    # get only the time component and convert it to string format.
    watch_one_df['timestamp_str'] = watch_one_df['timestamp'].dt.time.astype(str)

    watch_one_df.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/" +session_id+"/"+ "sorted_ts_" + watchID + ".csv")
    
    return watch_one_df 


def subset_activity_data(SO3_df,act_st,act_stop,date_id,watchID):
    # Match start TS id it exists.
    dt_start = datetime.strptime(act_st.to_string().split(" ")[4], '%I:%M').time()
    print("dt_start: ",dt_start.strftime('%I:%M'))
    # Check if any timestamps start with "10:21"
    timestamps_exist = SO3_df['timestamp'].dt.strftime('%I:%M').str.startswith(dt_start.strftime('%I:%M'))
    print("timestamps_exist: ",timestamps_exist.any())
    print("act_stop: ",act_stop.to_string().split(" ")[4])
    dt_end = datetime.strptime(act_stop.to_string().split(" ")[4], '%I:%M').time()
    print("end TS - ",dt_end.strftime('%I:%M'))
    end_ts_exists = SO3_df['timestamp'].dt.strftime('%I:%M').str.startswith(dt_end.strftime('%I:%M'))
    # if timestamps exists then.
    if timestamps_exist.any() and end_ts_exists.any():
        first_index = timestamps_exist.idxmax()
        last_index = end_ts_exists[::-1].idxmax()
        print("last_index",SO3_df.iloc[last_index])
        # Subset the Activity data by index.
        # +1 is added to include the end index
        activity_subset = SO3_df.iloc[first_index:last_index+1]
        return activity_subset # write this to a directory.
    else:
        print("No match found for START AND END TIME")
        return pd.DataFrame() # return empty data frame.


# Change these to process rach session sperately. 
session_id = "Session3_DC_05_07_2023"
timings_id = "ActivityTimingsSession3_05_07_23.csv"

root_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/" +session_id +"/"
# avaliable watch IDs from which data were collected.
watch_list = os.listdir(root_path)
# Read the activity timings.
df_activity_timings = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/"+timings_id)
# df_activity_timings
act_st1 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A1"]["ActivityStart"]
act_stop1 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A1"]["ActivityStop"]

act_st2 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A2"]["ActivityStart"]
act_stop2 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A2"]["ActivityStop"]

act_st3 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A3"]["ActivityStart"]
act_stop3 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A3"]["ActivityStop"]

act_st4 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A4"]["ActivityStart"]
act_stop4 = df_activity_timings[df_activity_timings["ActivityNumber"]=="A4"]["ActivityStop"]

file_write_root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/"
for watchid in watch_list:
    if watchid.startswith("SSW"):
        if watchid !="results_out":
            print("watchid: ",watchid)
            watch_one_df = read_all_sub_files_in_watch(watchid)  
            # subset_activity_data(SO3_df,act_st,act_stop,date_id,watchID)
            activity_subset1 = subset_activity_data(watch_one_df,act_st1,act_stop1,session_id,watchid)
            activity_subset2 = subset_activity_data(watch_one_df,act_st2,act_stop2,session_id,watchid)
            activity_subset3 = subset_activity_data(watch_one_df,act_st3,act_stop3,session_id,watchid)
            activity_subset4 = subset_activity_data(watch_one_df,act_st4,act_stop4,session_id,watchid)
            if not os.path.exists(file_write_root +session_id+"/A1/"):
                os.makedirs(file_write_root +session_id+"/A1/")
            if not os.path.exists(file_write_root +session_id+"/A2/"):
                os.makedirs(file_write_root +session_id+"/A2/")
            if not os.path.exists(file_write_root +session_id+"/A3/"):
                os.makedirs(file_write_root +session_id+"/A3/")
            if not os.path.exists(file_write_root +session_id+"/A4/"):
                os.makedirs(file_write_root +session_id+ "/A4/")
            activity_subset1.to_csv(file_write_root +session_id +"/A1/" + watchid + "_" + "A1.csv",index=0)
            print("activity_subset1 ============================ ",activity_subset1.head())
            activity_subset2.to_csv(file_write_root +session_id +"/A2/" + watchid + "_" + "A2.csv",index=0)
            print("activity_subset2 ============================ ",activity_subset2.head())
            activity_subset3.to_csv(file_write_root +session_id + "/A3/" +watchid + "_" + "A3.csv",index=0)
            print("activity_subset3 ============================ ",activity_subset3.head())
            activity_subset4.to_csv(file_write_root +session_id + "/A4/" +watchid + "_" + "A4.csv",index=0)
            print("activity_subset4 ============================ ",activity_subset4.head())
            
            


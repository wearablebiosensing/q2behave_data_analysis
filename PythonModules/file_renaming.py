import os
import json
# 'result_P9_SSW4WearOS03_A2.csv'

filenamelist1 = [
"P5_Left_SSW4WearOS07",
"P5_Right_SSW4WearOS08",
"P6_Left_SSW4WearOS03",
"P6_Right_SSW4WearOS04",
"P7_Left_SSW4WearOS07",
"P7_Left_SSW4WearOS01",
"P7_Right_SSW4WearOS08",
"P8_Left_SSW4WearOS03",
"P8_Right_SSW4WearOS04",
"P9_Left_SSW4WearOS03",
"P9_Right_SSW4WearOS04",
"P10_Left_SSW4WearOS02",
"P10_Right_SSW4WearOS05", 
"P10_Right_SSW4WearOS01",
"P11_Left_SSW4WearOS01",
"P11_Right_SSW4WearOS08",
"P13_Left_SSW4WearOS03",
"P13_Right_SSW4WearOS04",
"P14_Left_SSW4WearOS03",
"P14_Right_SSW4WearOS04",
"P14_Right_SSW4WearOS02",
"P15_Left_SSW4WearOS07",
"P15_Right_SSW4WearOS08",
"P16_Left_SSW4WearOS01",
"P16_Right_SSW4WearOS08",
"P12_Left_SSW4WearOS03",
"P12_Right_SSW4WearOS04",
"P18_Left_SSW4WearOS03",
"P18_Right_SSW4WearOS04",
"P19_Left_SSW4WearOS03",
"P19_Right_SSW4WearOS04",
"P20_Left_SSW4WearOS07",
"P20_Right_SSW4WearOS08",
"P21_Left_SSW4WearOS05",
"P21_Right_SSW4WearOS06",
"P22_Left_SSW4WearOS01",
"P22_Right_SSW4WearOS08",
"P23_Left_SSW4WearOS03",
"P23_Right_SSW4WearOS04",
"P24_Left_SSW4WearOS02",
"P24_Right_SSW4WearOS06"
]

folder_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/SyncedFilesAll"
filenamelist2 = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# print(csv_files)



import re
import shutil
import os

def extract_details(filename):
    # print("extract_details():/ filename ", filename)
    # print("Participant ID: ",filename.split("_")[1], "Watch ID : ",filename.split("_")[2])
    participant_ID = filename.split("_")[1]
    watch_ID = filename.split("_")[2][-3:]
    print("watch_ID) sub:",watch_ID,type(watch_ID))
    return participant_ID, watch_ID
def create_mapping_dict(filenamelist1):
    # Create a dictionary to map (participant, watch) -> full name from filenamelist1
    mapping = {}
    for name in filenamelist1:
        print("mapping(): name",name) 
        parts = name.split("_")
        participant, watch = parts[0], parts[2][-3:]
        print("Participant, Watch mapping(): ",participant, watch)
        mapping[(participant, watch)] = name
    print("mapping:",mapping)
    mapping_filename = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/q2behave/PythonModules/mapping.json"
    # # Option 1: Write JSON data to a file using json.dump()
    # with open(mapping_filename, 'w') as json_file:
    #     json.dump(mapping, json_file)
    return mapping

source_dir = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/SyncedFilesAll"
target_dir = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/RenamedData"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

missed_files = []
copied_files = []

mapping = create_mapping_dict(filenamelist1)
# Rename filenames in filenamelist2 based on the mapping and copy to new directory
for file in filenamelist2:
    participant, watch = extract_details(file)
    print("file: ",file)
    print("participant, watch: ",participant, watch)
    if (participant, watch) in mapping:
        base_name = mapping[(participant, watch)]
        activity = file.split("_")[-1]
        new_base_name = base_name.replace(f"_{watch}", "")  # Removing the watch ID
        print("new_base_name: ",new_base_name)
        result_string = new_base_name.replace(new_base_name.split("_")[2], "")

        new_name = f"{result_string}{activity}"

        # Copy the file to the new directory if it was renamed
        shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, new_name))
        copied_files.append(file)
    else:
        print("No Match Found! ===============================")
        # If no match was found, add the file to the missed_files list
        missed_files.append(file)

all_files_in_source = set(os.listdir(source_dir))
not_copied_files = all_files_in_source - set(copied_files)

print("Missed Files from filenamelist2:")
print("\n".join(missed_files))

print("\nFiles in source_dir that were not copied:")
print("\n".join(not_copied_files))

































































































































































































































































































































































































































































































































































































































































































































































































































































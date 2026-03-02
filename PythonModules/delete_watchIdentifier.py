import os
import pandas as pd

# Define the folder path containing the CSV files
folder_path = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/SyncedFilesAll'

# List all the files in the folder
files = os.listdir(folder_path)

# Loop through each file
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path,low_memory=False)

        # Check if the "WatchID" column exists in the DataFrame
        if 'WatchID' in df.columns:
            # Delete the "WatchID" column
            df.drop(columns=['WatchID'], inplace=True)

            # Write the modified DataFrame back to the same file
            df.to_csv(file_path, index=False)

            print(f'Removed "WatchID" column from {file}')
        else:
            print(f'"WatchID" column not found in {file}')

print('All files processed.')

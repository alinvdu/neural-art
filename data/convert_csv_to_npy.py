import os
import pandas as pd
import numpy as np

def convert_csv_to_npy(source_folder, target_folder):
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file in os.listdir(source_folder):
        print(file)
        if file.endswith(".csv"):
            file_path = os.path.join(source_folder, file)
            df = pd.read_csv(file_path).transpose()
            npy_file_name = os.path.splitext(file)[0] + '.npy'
            npy_file_path = os.path.join(target_folder, npy_file_name)
            np.save(npy_file_path, df.to_numpy())

# Replace 'source_folder_path' with the path to the folder containing your CSV files
source_folder_path = '../test_data/processed/csv_raw/experiment2/trial1/cleaned'
# Folder to save the .npy files
npy_folder_path = os.path.join(source_folder_path, 'npy')
convert_csv_to_npy(source_folder_path, npy_folder_path)
